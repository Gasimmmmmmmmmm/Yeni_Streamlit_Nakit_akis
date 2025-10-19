
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import base64
import tempfile
import os
warnings.filterwarnings('ignore')

# TensorFlow import
try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow yüklü değil. LSTM özelliği devre dışı.")

# Sayfa Ayarları
st.set_page_config(
    page_title="💰 Nakit Akış Sistemi",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #2c3e50;
        padding-bottom: 1rem;
    }
    h2 {
        color: #34495e;
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL AYARLARI ====================
MODEL_FILE_PATH = "/content/model_embedded.txt"  # Colab için
# MODEL_FILE_PATH = "model_embedded.txt"  # Lokal için

# ==================== SESSION STATE ====================
if 'sistem' not in st.session_state:
    st.session_state.sistem = None
    st.session_state.initialized = False

class NakitAkisYonetimi:
    def __init__(self):
        self.hareketler = pd.DataFrame(columns=['Tarih', 'Aciklama', 'Kategori', 'Tutar', 'Tip', 'Hesap'])
        self.hesaplar = {}
        self.lstm_model = None
        self.model_yuklendi = False
        
        if TENSORFLOW_AVAILABLE:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self._embedded_model_yukle()

    def _embedded_model_yukle(self):
        """Gömülü modeli dosyadan otomatik yükler"""
        try:
            if not os.path.exists(MODEL_FILE_PATH):
                st.info(f"📁 Model dosyası bulunamadı: {MODEL_FILE_PATH}")
                return
            
            with open(MODEL_FILE_PATH, 'r') as f:
                model_base64 = f.read().strip()
            
            if not model_base64 or len(model_base64) < 100:
                st.warning("⚠️ Model dosyası boş veya geçersiz.")
                return
            
            model_bytes = base64.b64decode(model_base64)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(model_bytes)
                tmp_path = tmp_file.name
            
            self.lstm_model = tf.keras.models.load_model(tmp_path)
            self.model_yuklendi = True
            os.unlink(tmp_path)
            
            st.success(f"✅ LSTM Model yüklendi! Katmanlar: {len(self.lstm_model.layers)}")
            
        except Exception as e:
            st.error(f"⚠️ Model yükleme hatası: {str(e)}")
            self.model_yuklendi = False

    def csv_yukle(self, dosya, hesap_adi, baslangic_bakiye, tarih_sutun, tutar_sutun, aciklama_sutun):
        """CSV dosyasını yükler ve işler"""
        try:
            df = pd.read_csv(dosya, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()

            if tarih_sutun not in df.columns:
                return False, f"❌ '{tarih_sutun}' sütunu bulunamadı. Mevcut: {', '.join(df.columns)}", pd.DataFrame()

            yeni_df = pd.DataFrame()
            yeni_df['Tarih'] = pd.to_datetime(df[tarih_sutun], errors='coerce')
            yeni_df['Tutar'] = pd.to_numeric(df[tutar_sutun], errors='coerce')
            yeni_df['Aciklama'] = df[aciklama_sutun] if aciklama_sutun and aciklama_sutun in df.columns else 'İşlem'

            yeni_df['Tip'] = yeni_df['Tutar'].apply(lambda x: 'Giris' if x > 0 else 'Cikis')
            yeni_df['Tutar'] = yeni_df['Tutar'].abs()
            yeni_df['Kategori'] = yeni_df['Aciklama'].apply(self._kategori_tahmin)
            yeni_df['Hesap'] = hesap_adi

            yeni_df = yeni_df.dropna(subset=['Tarih', 'Tutar'])

            if len(self.hareketler) == 0:
                self.hareketler = yeni_df.copy()
            else:
                self.hareketler = pd.concat([self.hareketler, yeni_df], ignore_index=True)

            self.hareketler = self.hareketler.sort_values('Tarih').reset_index(drop=True)
            self.hesaplar[hesap_adi] = float(baslangic_bakiye)

            ozet_df = yeni_df.head(10)[['Tarih', 'Aciklama', 'Tutar', 'Tip', 'Kategori']].copy()

            return True, f"✅ {hesap_adi} yüklendi! {len(yeni_df)} işlem eklendi.", ozet_df

        except Exception as e:
            return False, f"❌ Hata: {str(e)}", pd.DataFrame()

    def _kategori_tahmin(self, aciklama):
        """Açıklamaya göre kategori tahmin eder"""
        aciklama_lower = str(aciklama).lower()

        if any(x in aciklama_lower for x in ['maas', 'maaş', 'salary']):
            return 'Maaş'
        elif any(x in aciklama_lower for x in ['kira', 'rent']):
            return 'Kira'
        elif any(x in aciklama_lower for x in ['elektrik', 'su', 'gaz', 'fatura', 'bill']):
            return 'Fatura'
        elif any(x in aciklama_lower for x in ['market', 'migros', 'carrefour']):
            return 'Market'
        elif any(x in aciklama_lower for x in ['transfer', 'havale']):
            return 'Transfer'
        else:
            return 'Diger'

    def manuel_islem_ekle(self, tarih, aciklama, kategori, tutar, tip, hesap):
        """Manuel nakit hareketi ekler"""
        try:
            yeni_hareket = pd.DataFrame([{
                'Tarih': pd.to_datetime(tarih),
                'Aciklama': aciklama,
                'Kategori': kategori,
                'Tutar': float(tutar),
                'Tip': tip,
                'Hesap': hesap
            }])

            if len(self.hareketler) == 0:
                self.hareketler = yeni_hareket.copy()
            else:
                self.hareketler = pd.concat([self.hareketler, yeni_hareket], ignore_index=True)

            self.hareketler = self.hareketler.sort_values('Tarih').reset_index(drop=True)

            return True, f"✅ Eklendi: {aciklama} - {tutar} TL ({tip})"
        except Exception as e:
            return False, f"❌ Hata: {str(e)}"

    def analiz_yap(self, baslangic_tarihi, bitis_tarihi, baslangic_bakiye, buffer_tutar):
        """Kapsamlı nakit akış analizi yapar"""
        try:
            if len(self.hareketler) == 0:
                return None, None, None, "⚠️ Henüz işlem yüklenmedi!"

            baslangic = pd.to_datetime(baslangic_tarihi)
            bitis = pd.to_datetime(bitis_tarihi)

            tarih_araligi = pd.date_range(start=baslangic, end=bitis, freq='D')
            bakiye_df = pd.DataFrame({'Tarih': tarih_araligi})
            bakiye_df['Bakiye'] = float(baslangic_bakiye)

            ilgili_hareketler = self.hareketler[
                (self.hareketler['Tarih'] >= baslangic) &
                (self.hareketler['Tarih'] <= bitis)
            ].copy()

            for idx in range(len(bakiye_df)):
                tarih = bakiye_df.loc[idx, 'Tarih']

                if idx > 0:
                    bakiye_df.loc[idx, 'Bakiye'] = bakiye_df.loc[idx-1, 'Bakiye']

                gunun_hareketleri = ilgili_hareketler[
                    ilgili_hareketler['Tarih'].dt.date == tarih.date()
                ]

                for _, hareket in gunun_hareketleri.iterrows():
                    if hareket['Tip'] == 'Giris':
                        bakiye_df.loc[idx, 'Bakiye'] += hareket['Tutar']
                    else:
                        bakiye_df.loc[idx, 'Bakiye'] -= hareket['Tutar']

            bakiye_df['Yatirilabilir'] = bakiye_df['Bakiye'].apply(lambda x: max(0, x - buffer_tutar))
            bakiye_df['Durum'] = bakiye_df['Bakiye'].apply(
                lambda x: '🟢 Fazla' if x > buffer_tutar else ('🟡 Normal' if x >= 0 else '🔴 Acik')
            )

            oneriler_df = self._yatirim_onerileri_olustur(bakiye_df)
            grafik = self._grafik_olustur(bakiye_df, ilgili_hareketler)

            toplam_giris = ilgili_hareketler[ilgili_hareketler['Tip'] == 'Giris']['Tutar'].sum()
            toplam_cikis = ilgili_hareketler[ilgili_hareketler['Tip'] == 'Cikis']['Tutar'].sum()
            net_akis = toplam_giris - toplam_cikis
            min_bakiye = bakiye_df['Bakiye'].min()
            max_bakiye = bakiye_df['Bakiye'].max()

            ozet = {
                'toplam_giris': toplam_giris,
                'toplam_cikis': toplam_cikis,
                'net_akis': net_akis,
                'min_bakiye': min_bakiye,
                'max_bakiye': max_bakiye,
                'acik_gun': len(bakiye_df[bakiye_df['Bakiye'] < 0])
            }

            display_df = bakiye_df.copy()
            display_df['Tarih'] = display_df['Tarih'].dt.strftime('%Y-%m-%d')
            display_df['Bakiye'] = display_df['Bakiye'].apply(lambda x: f"{x:,.2f}")
            display_df['Yatirilabilir'] = display_df['Yatirilabilir'].apply(lambda x: f"{x:,.2f}")

            return display_df, oneriler_df, grafik, ozet

        except Exception as e:
            return None, None, None, f"❌ Hata: {str(e)}"

    def _yatirim_onerileri_olustur(self, bakiye_df):
        """Yatırım fırsatlarını tespit eder"""
        oneriler = []

        i = 0
        while i < len(bakiye_df):
            yatirilabilir = bakiye_df.iloc[i]['Yatirilabilir']

            if yatirilabilir > 100:
                gun_sayisi = 1
                min_tutar = yatirilabilir

                for j in range(i+1, len(bakiye_df)):
                    if bakiye_df.iloc[j]['Yatirilabilir'] > 0:
                        gun_sayisi += 1
                        min_tutar = min(min_tutar, bakiye_df.iloc[j]['Yatirilabilir'])
                    else:
                        break

                if gun_sayisi >= 1:
                    yillik_faiz = 0.45
                    gunluk_faiz = yillik_faiz / 365
                    tahmini_getiri = min_tutar * (gunluk_faiz * gun_sayisi)

                    oneriler.append({
                        'Baslangic': bakiye_df.iloc[i]['Tarih'].strftime('%Y-%m-%d'),
                        'Gun': gun_sayisi,
                        'Tutar': f"{min_tutar:,.2f} TL",
                        'Tahmini Getiri': f"{tahmini_getiri:,.2f} TL",
                        'Yatirim Araci': self._yatirim_araci_sec(gun_sayisi),
                        'Yillik Getiri': f"%{(tahmini_getiri/min_tutar)*(365/gun_sayisi)*100:.2f}"
                    })

                i += gun_sayisi
            else:
                i += 1

        return pd.DataFrame(oneriler) if oneriler else pd.DataFrame()

    def _yatirim_araci_sec(self, gun):
        """Gün sayısına göre yatırım aracı önerir"""
        if gun <= 1:
            return "📊 Overnight Repo"
        elif gun <= 7:
            return "📈 Haftalik Repo"
        elif gun <= 30:
            return "🏦 Vadeli Mevduat (1 Ay)"
        elif gun <= 90:
            return "💰 Vadeli Mevduat (3 Ay)"
        else:
            return "💎 Vadeli Mevduat (6 Ay+)"

    def _grafik_olustur(self, bakiye_df, hareketler_df):
        """Gelişmiş interaktif grafik oluşturur"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('💰 Günlük Bakiye Akışı', '💵 Yatırılabilir Tutar', '📊 Giriş/Çıkış'),
            vertical_spacing=0.12,
            row_heights=[0.4, 0.3, 0.3]
        )

        fig.add_trace(
            go.Scatter(
                x=bakiye_df['Tarih'],
                y=bakiye_df['Bakiye'],
                mode='lines+markers',
                name='Bakiye',
                line=dict(color='#3498db', width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.1)',
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Bakiye: %{y:,.2f} TL<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1, row=1, col=1)

        fig.add_trace(
            go.Bar(
                x=bakiye_df['Tarih'],
                y=bakiye_df['Yatirilabilir'],
                name='Yatirilabilir',
                marker_color='#2ecc71',
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Yatırılabilir: %{y:,.2f} TL<extra></extra>'
            ),
            row=2, col=1
        )

        if len(hareketler_df) > 0:
            gunluk_giris = hareketler_df[hareketler_df['Tip'] == 'Giris'].groupby(
                hareketler_df['Tarih'].dt.date
            )['Tutar'].sum()
            gunluk_cikis = hareketler_df[hareketler_df['Tip'] == 'Cikis'].groupby(
                hareketler_df['Tarih'].dt.date
            )['Tutar'].sum()

            if len(gunluk_giris) > 0:
                fig.add_trace(
                    go.Bar(
                        x=gunluk_giris.index,
                        y=gunluk_giris.values,
                        name='Giris',
                        marker_color='#27ae60',
                        hovertemplate='<b>%{x}</b><br>Giriş: %{y:,.2f} TL<extra></extra>'
                    ),
                    row=3, col=1
                )

            if len(gunluk_cikis) > 0:
                fig.add_trace(
                    go.Bar(
                        x=gunluk_cikis.index,
                        y=-gunluk_cikis.values,
                        name='Cikis',
                        marker_color='#e74c3c',
                        hovertemplate='<b>%{x}</b><br>Çıkış: %{y:,.2f} TL<extra></extra>'
                    ),
                    row=3, col=1
                )

        fig.update_layout(
            height=1000,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )

        fig.update_xaxes(title_text="Tarih", row=3, col=1)
        fig.update_yaxes(title_text="Bakiye (TL)", row=1, col=1)
        fig.update_yaxes(title_text="Tutar (TL)", row=2, col=1)
        fig.update_yaxes(title_text="Tutar (TL)", row=3, col=1)

        return fig

    def gunluk_bakiye_serisi_olustur(self, baslangic_bakiye):
        """Günlük bakiye serisini oluşturur"""
        if len(self.hareketler) == 0:
            return None

        min_tarih = self.hareketler['Tarih'].min()
        max_tarih = self.hareketler['Tarih'].max()

        tarih_araligi = pd.date_range(start=min_tarih, end=max_tarih, freq='D')
        bakiye_serisi = pd.DataFrame({'Tarih': tarih_araligi, 'Bakiye': float(baslangic_bakiye)})

        for idx in range(len(bakiye_serisi)):
            if idx > 0:
                bakiye_serisi.loc[idx, 'Bakiye'] = bakiye_serisi.loc[idx-1, 'Bakiye']

            tarih = bakiye_serisi.loc[idx, 'Tarih']
            gunun_hareketleri = self.hareketler[self.hareketler['Tarih'].dt.date == tarih.date()]

            for _, hareket in gunun_hareketleri.iterrows():
                if hareket['Tip'] == 'Giris':
                    bakiye_serisi.loc[idx, 'Bakiye'] += hareket['Tutar']
                else:
                    bakiye_serisi.loc[idx, 'Bakiye'] -= hareket['Tutar']

        return bakiye_serisi

    def lstm_tahmin_yap(self, tahmin_gun_sayisi, baslangic_bakiye, lookback=30):
        """LSTM ile tahmin yapar"""
        if not TENSORFLOW_AVAILABLE:
            return None, None, "❌ TensorFlow yüklü değil!"

        try:
            if not self.model_yuklendi or self.lstm_model is None:
                return None, None, f"❌ Model yüklenemedi! {MODEL_FILE_PATH} kontrol edin."

            if len(self.hareketler) == 0:
                return None, None, "❌ Veri yok! Önce CSV yükleyin."

            bakiye_serisi = self.gunluk_bakiye_serisi_olustur(baslangic_bakiye)

            if bakiye_serisi is None or len(bakiye_serisi) < lookback:
                return None, None, f"❌ En az {lookback} gün veri gerekli!"

            bakiye_degerleri = bakiye_serisi['Bakiye'].values.reshape(-1, 1)
            normalized_data = self.scaler.fit_transform(bakiye_degerleri)

            son_veri = normalized_data[-lookback:]

            tahminler = []
            mevcut_veri = son_veri.copy()

            for _ in range(tahmin_gun_sayisi):
                X_test = mevcut_veri.reshape(1, lookback, 1)
                tahmin = self.lstm_model.predict(X_test, verbose=0)
                tahminler.append(tahmin[0, 0])
                mevcut_veri = np.append(mevcut_veri[1:], tahmin)

            tahminler = np.array(tahminler).reshape(-1, 1)
            tahminler_gercek = self.scaler.inverse_transform(tahminler)

            son_tarih = bakiye_serisi['Tarih'].max()
            tahmin_tarihleri = pd.date_range(
                start=son_tarih + timedelta(days=1),
                periods=tahmin_gun_sayisi,
                freq='D'
            )

            tahmin_df = pd.DataFrame({
                'Tarih': tahmin_tarihleri,
                'Tahmin_Bakiye': tahminler_gercek.flatten()
            })

            grafik = self._lstm_tahmin_grafik(bakiye_serisi, tahmin_df)

            min_tahmin = tahminler_gercek.min()
            max_tahmin = tahminler_gercek.max()
            ort_tahmin = tahminler_gercek.mean()

            ozet = {
                'tahmin_suresi': tahmin_gun_sayisi,
                'ortalama': ort_tahmin,
                'minimum': min_tahmin,
                'maximum': max_tahmin,
                'risk': 'Yüksek' if min_tahmin < 0 else 'Düşük'
            }

            display_df = tahmin_df.copy()
            display_df['Tarih'] = display_df['Tarih'].dt.strftime('%Y-%m-%d')
            display_df['Tahmin_Bakiye'] = display_df['Tahmin_Bakiye'].apply(lambda x: f"{x:,.2f}")
            display_df['Durum'] = tahmin_df['Tahmin_Bakiye'].apply(
                lambda x: '🟢 Pozitif' if x >= 0 else '🔴 Negatif'
            )

            return display_df, grafik, ozet

        except Exception as e:
            return None, None, f"❌ Hata: {str(e)}"

    def _lstm_tahmin_grafik(self, gercek_veri, tahmin_veri):
        """LSTM grafik oluşturur"""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=gercek_veri['Tarih'],
                y=gercek_veri['Bakiye'],
                mode='lines',
                name='Gerçek Bakiye',
                line=dict(color='#3498db', width=2)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=tahmin_veri['Tarih'],
                y=tahmin_veri['Tahmin_Bakiye'],
                mode='lines+markers',
                name='LSTM Tahmini',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                marker=dict(size=6)
            )
        )

        fig.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="Kritik")

        fig.update_layout(
            title="🔮 LSTM Nakit Akış Tahmini",
            xaxis_title="Tarih",
            yaxis_title="Bakiye (TL)",
            height=600,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )

        return fig

    def tum_islemleri_goster(self):
        """Tüm işlemleri gösterir"""
        if len(self.hareketler) == 0:
            return pd.DataFrame()

        df = self.hareketler.copy()
        df['Tarih'] = df['Tarih'].dt.strftime('%Y-%m-%d')
        df = df.sort_values('Tarih', ascending=False)
        return df

# ==================== STREAMLIT UI ====================

def main():
    # Header
    st.title("💰 Nakit Akış Yönetim Sistemi + 🔮 LSTM Tahmin")
    st.markdown("### Nakit akışınızı takip edin, AI ile geleceği öngörün!")
    
    # Sistem initialize
    if not st.session_state.initialized:
        st.session_state.sistem = NakitAkisYonetimi()
        st.session_state.initialized = True
    
    sistem = st.session_state.sistem
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # Model durumu
        st.subheader("🤖 Model Durumu")
        if TENSORFLOW_AVAILABLE and sistem.model_yuklendi:
            st.success(f"✅ Model Hazır\n\nKatmanlar: {len(sistem.lstm_model.layers)}")
        elif TENSORFLOW_AVAILABLE:
            st.warning(f"⚠️ Model Yok\n\n{MODEL_FILE_PATH}")
        else:
            st.error("❌ TensorFlow Yok")
        
        st.divider()
        
        # İstatistikler
        st.subheader("📊 İstatistikler")
        if len(sistem.hareketler) > 0:
            st.metric("Toplam İşlem", len(sistem.hareketler))
            st.metric("Hesap Sayısı", len(sistem.hesaplar))
            st.metric("Tarih Aralığı", 
                     f"{sistem.hareketler['Tarih'].min().strftime('%Y-%m-%d')}\n-\n{sistem.hareketler['Tarih'].max().strftime('%Y-%m-%d')}")
        else:
            st.info("Henüz veri yok")
    
    # Ana İçerik - Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 CSV Yükle",
        "➕ Manuel İşlem", 
        "📊 Analiz",
        "🔮 LSTM Tahmin",
        "📋 Tüm İşlemler",
        "❓ Yardım"
    ])
    
    # TAB 1: CSV YÜKLE
    with tab1:
        st.header("📤 CSV Dosyası Yükle")
        st.markdown("Banka ekstrenizi yükleyin ve otomatik olarak işlensin.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_file = st.file_uploader("CSV Dosyası Seçin", type=['csv'], key="csv_upload")
            hesap_adi = st.text_input("Hesap Adı", value="Ana Hesap")
            baslangic_bakiye = st.number_input("Başlangıç Bakiyesi (TL)", value=10000.0, step=100.0)
        
        with col2:
            tarih_sutun = st.text_input("Tarih Sütunu Adı", value="Tarih")
            tutar_sutun = st.text_input("Tutar Sütunu Adı", value="Tutar")
            aciklama_sutun = st.text_input("Açıklama Sütunu Adı", value="Aciklama")
        
        if st.button("📥 CSV Yükle ve İşle", type="primary", use_container_width=True):
            if csv_file is not None:
                with st.spinner("Yükleniyor..."):
                    success, mesaj, ozet_df = sistem.csv_yukle(
                        csv_file, hesap_adi, baslangic_bakiye,
                        tarih_sutun, tutar_sutun, aciklama_sutun
                    )
                    
                    if success:
                        st.success(mesaj)
                        if not ozet_df.empty:
                            st.subheader("📋 İlk 10 İşlem Önizlemesi")
                            st.dataframe(ozet_df, use_container_width=True)
                    else:
                        st.error(mesaj)
            else:
                st.warning("⚠️ Lütfen bir CSV dosyası seçin")
    
    # TAB 2: MANUEL İŞLEM
    with tab2:
        st.header("➕ Manuel İşlem Ekle")
        st.markdown("Nakit hareketi manuel olarak ekleyin.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            m_tarih = st.date_input("Tarih", value=datetime.now())
            m_aciklama = st.text_input("Açıklama", placeholder="Örn: Kira ödemesi")
            m_kategori = st.selectbox("Kategori", ["Maaş", "Kira", "Fatura", "Market", "Transfer", "Diger"])
        
        with col2:
            m_tutar = st.number_input("Tutar (TL)", value=0.0, step=10.0)
            m_tip = st.radio("İşlem Tipi", ["Giris", "Cikis"], horizontal=True)
            m_hesap = st.text_input("Hesap", value="Kasa")
        
        if st.button("➕ İşlem Ekle", type="primary", use_container_width=True):
            if m_aciklama and m_tutar > 0:
                success, mesaj = sistem.manuel_islem_ekle(
                    m_tarih, m_aciklama, m_kategori, m_tutar, m_tip, m_hesap
                )
                if success:
                    st.success(mesaj)
                    st.balloons()
                else:
                    st.error(mesaj)
            else:
                st.warning("⚠️ Lütfen tüm alanları doldurun")
    
    # TAB 3: ANALİZ
    with tab3:
        st.header("📊 Nakit Akış Analizi")
        
        if len(sistem.hareketler) == 0:
            st.warning("⚠️ Önce CSV yükleyin veya manuel işlem ekleyin")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                a_baslangic = st.date_input("Başlangıç Tarihi", value=datetime.now())
            with col2:
                a_bitis = st.date_input("Bitiş Tarihi", value=datetime.now() + timedelta(days=30))
            with col3:
                a_baslangic_bakiye = st.number_input("Başlangıç Bakiyesi", value=10000.0, step=100.0)
            with col4:
                a_buffer = st.number_input("Buffer Tutarı", value=1000.0, step=100.0)
            
            if st.button("🔍 ANALİZ YAP", type="primary", use_container_width=True):
                with st.spinner("Analiz yapılıyor..."):
                    bakiye_df, oneriler_df, grafik, ozet = sistem.analiz_yap(
                        a_baslangic, a_bitis, a_baslangic_bakiye, a_buffer
                    )
                    
                    if grafik is not None:
                        # Özet Kartlar
                        st.subheader("📈 Özet Bilgiler")
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        
                        with col1:
                            st.metric("Toplam Giriş", f"{ozet['toplam_giris']:,.0f} TL", 
                                     delta_color="normal")
                        with col2:
                            st.metric("Toplam Çıkış", f"{ozet['toplam_cikis']:,.0f} TL",
                                     delta_color="inverse")
                        with col3:
                            st.metric("Net Akış", f"{ozet['net_akis']:,.0f} TL",
                                     delta=f"{ozet['net_akis']:,.0f}" if ozet['net_akis'] >= 0 else f"-{abs(ozet['net_akis']):,.0f}")
                        with col4:
                            st.metric("Min Bakiye", f"{ozet['min_bakiye']:,.0f} TL")
                        with col5:
                            st.metric("Max Bakiye", f"{ozet['max_bakiye']:,.0f} TL")
                        with col6:
                            st.metric("Açık Gün", f"{ozet['acik_gun']} gün")
                        
                        st.divider()
                        
                        # Grafik
                        st.subheader("📊 İnteraktif Grafikler")
                        st.plotly_chart(grafik, use_container_width=True)
                        
                        st.divider()
                        
                        # Tablolar
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("💰 Günlük Bakiye")
                            st.dataframe(bakiye_df, use_container_width=True, height=400)
                        
                        with col2:
                            st.subheader("💎 Yatırım Önerileri")
                            if not oneriler_df.empty:
                                st.dataframe(oneriler_df, use_container_width=True, height=400)
                            else:
                                st.info("Yatırım fırsatı bulunamadı")
                    else:
                        st.error(ozet)
    
    # TAB 4: LSTM TAHMİN
    with tab4:
        st.header("🔮 LSTM ile AI Tahmin")
        
        if not TENSORFLOW_AVAILABLE:
            st.error("❌ TensorFlow yüklü değil! `pip install tensorflow`")
        elif not sistem.model_yuklendi:
            st.warning(f"⚠️ Model yüklenemedi. {MODEL_FILE_PATH} dosyasını kontrol edin.")
        elif len(sistem.hareketler) == 0:
            st.warning("⚠️ Önce CSV yükleyin veya manuel işlem ekleyin")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tahmin_gun = st.slider("Tahmin Süresi (Gün)", 1, 90, 30)
            with col2:
                tahmin_bakiye = st.number_input("Başlangıç Bakiyesi", value=10000.0, step=100.0, key="tahmin_bakiye")
            with col3:
                lookback = st.slider("Lookback Period", 7, 60, 30)
            
            if st.button("🔮 TAHMİN YAP", type="primary", use_container_width=True):
                with st.spinner("AI modeli çalışıyor..."):
                    tahmin_df, tahmin_grafik, tahmin_ozet = sistem.lstm_tahmin_yap(
                        tahmin_gun, tahmin_bakiye, lookback
                    )
                    
                    if tahmin_grafik is not None:
                        # Özet Kartlar
                        st.subheader("📈 Tahmin Özeti")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Tahmin Süresi", f"{tahmin_ozet['tahmin_suresi']} gün")
                        with col2:
                            st.metric("Ort. Bakiye", f"{tahmin_ozet['ortalama']:,.0f} TL")
                        with col3:
                            st.metric("Min Bakiye", f"{tahmin_ozet['minimum']:,.0f} TL")
                        with col4:
                            st.metric("Max Bakiye", f"{tahmin_ozet['maximum']:,.0f} TL")
                        with col5:
                            risk_color = "🔴" if tahmin_ozet['risk'] == 'Yüksek' else "🟢"
                            st.metric("Açık Riski", f"{risk_color} {tahmin_ozet['risk']}")
                        
                        st.divider()
                        
                        # Grafik
                        st.subheader("📊 Tahmin Grafiği")
                        st.plotly_chart(tahmin_grafik, use_container_width=True)
                        
                        st.divider()
                        
                        # Tablo
                        st.subheader("📋 Tahmin Detayları")
                        st.dataframe(tahmin_df, use_container_width=True, height=400)
                    else:
                        st.error(tahmin_ozet)
    
    # TAB 5: TÜM İŞLEMLER
    with tab5:
        st.header("📋 Tüm İşlemler")
        
        if len(sistem.hareketler) == 0:
            st.info("Henüz işlem yok")
        else:
            # Filtreler
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filtre_tip = st.multiselect("İşlem Tipi", ["Giris", "Cikis"], default=["Giris", "Cikis"])
            with col2:
                filtre_kategori = st.multiselect("Kategori", 
                    sistem.hareketler['Kategori'].unique().tolist(),
                    default=sistem.hareketler['Kategori'].unique().tolist())
            with col3:
                filtre_hesap = st.multiselect("Hesap",
                    sistem.hareketler['Hesap'].unique().tolist(),
                    default=sistem.hareketler['Hesap'].unique().tolist())
            
            # Filtrelenmiş veri
            filtered_df = sistem.hareketler[
                (sistem.hareketler['Tip'].isin(filtre_tip)) &
                (sistem.hareketler['Kategori'].isin(filtre_kategori)) &
                (sistem.hareketler['Hesap'].isin(filtre_hesap))
            ].copy()
            
            filtered_df['Tarih'] = filtered_df['Tarih'].dt.strftime('%Y-%m-%d')
            filtered_df = filtered_df.sort_values('Tarih', ascending=False)
            
            st.subheader(f"📊 Toplam {len(filtered_df)} İşlem")
            st.dataframe(filtered_df, use_container_width=True, height=600)
            
            # İndir butonu
            csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV Olarak İndir",
                data=csv,
                file_name=f"nakit_akis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # TAB 6: YARDIM
    with tab6:
        st.header("❓ Kullanım Kılavuzu")
        
        with st.expander("📖 1. CSV Formatı", expanded=True):
            st.markdown("""
            CSV dosyanız şu formatta olmalı:
            
            ```csv
            Tarih,Tutar,Aciklama
            2024-10-01,5000,Maaş
            2024-10-02,-1200,Kira
            2024-10-05,-300,Market
            ```
            
            **Notlar:**
            - Pozitif tutar = Giriş
            - Negatif tutar = Çıkış
            - Tarih formatı: YYYY-MM-DD
            """)
        
        with st.expander("🚀 2. Kullanım Adımları"):
            st.markdown("""
            1. **CSV Yükle:** Banka ekstrenizi yükleyin
            2. **Manuel İşlem:** Gerekirse kasa hareketlerini ekleyin
            3. **Analiz:** Nakit akışınızı analiz edin
            4. **LSTM Tahmin:** AI ile gelecek tahmini yapın
            """)
        
        with st.expander("🤖 3. Model Kurulumu"):
            st.markdown(f"""
            **Model Dosyası Hazırlama:**
            
            ```python
            import base64
            
            with open('nakit_akis_lstm_final.h5', 'rb') as f:
                model_bytes = f.read()
                model_base64 = base64.b64encode(model_bytes).decode('utf-8')
            
            with open('model_embedded.txt', 'w') as f:
                f.write(model_base64)
            ```
            
            **Model Yükleme:**
            - Oluşan `model_embedded.txt` dosyasını şu konuma koyun:
            - `{MODEL_FILE_PATH}`
            - Uygulamayı yeniden başlatın
            """)
        
        with st.expander("💡 4. İpuçları"):
            st.markdown("""
            - En az 60-90 günlük veri kullanın
            - Lookback period'u 30 gün tutun
            - Düzenli veri güncelleyin
            - Buffer tutarını ihtiyacınıza göre ayarlayın
            - Yatırım önerileri %45 yıllık getiri üzerinden hesaplanır
            """)
        
        with st.expander("💰 5. Yatırım Araçları"):
            st.markdown("""
            | Süre | Araç | Özellik |
            |------|------|---------|
            | 1 gün | 📊 Overnight Repo | Günlük likidite |
            | 7 gün | 📈 Haftalık Repo | Kısa vade |
            | 30 gün | 🏦 Vadeli Mevduat (1 Ay) | Orta vade |
            | 90 gün | 💰 Vadeli Mevduat (3 Ay) | Uzun vade |
            | 90+ gün | 💎 Vadeli Mevduat (6 Ay+) | En yüksek getiri |
            """)
        
        with st.expander("🔧 6. Sorun Giderme"):
            st.markdown("""
            **"Model dosyası bulunamadı":**
            - Model dosyasının konumunu kontrol edin
            - Base64 dönüşümünü doğru yaptığınızdan emin olun
            
            **"Veri yok" hatası:**
            - Önce CSV yükleyin
            - En az 30 gün veri olmalı
            
            **"TensorFlow yüklü değil":**
            ```bash
            pip install tensorflow
            ```
            
            **Performans sorunları:**
            - Büyük veri setlerinde filtreleme kullanın
            - Tarih aralığını daraltın
            """)
        
        st.divider()
        
        st.info("""
        **📞 Destek & İletişim**
        
        Bu uygulama nakit akışı yönetimi ve AI destekli tahmin için geliştirilmiştir.
        Sorularınız için dokümantasyonu inceleyin.
        """)

if __name__ == "__main__":
    main()
