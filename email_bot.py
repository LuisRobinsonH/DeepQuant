# email_bot.py (VERSIÓN OUTLOOK / HOTMAIL)
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime

# --- CONFIGURACIÓN OUTLOOK ---
# Servidor oficial de Microsoft Office 365 / Outlook / Hotmail
SMTP_SERVER = "smtp.office365.com"
SMTP_PORT = 587
SENDER_EMAIL = "luchofigo98@hotmail.com"  # <--- TU EMAIL COMPLETO
SENDER_PASSWORD = "qdaamnygsuoczbob"     # <--- LA CONTRASEÑA DE APLICACIÓN QUE GENERASTE
DESTINATION_EMAIL = "luchofigo98@hotmail.com" # A donde te llega la alerta (puede ser el mismo)

def send_alert(ticker, action, price, probability, reason):
    try:
        subject = f"🚨 TITAN ALERT: {action} {ticker} @ ${price:.2f}"
        
        body = f"""
        🤖 TITAN LIVE SIGNAL (OUTLOOK)
        ==============================
        
        TICKER:   {ticker}
        ACCIÓN:   {action}
        PRECIO:   ${price:.2f}
        
        🧠 IA CONFIDENCE: {probability:.1%}
        🔍 RAZÓN: {reason}
        
        Hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = DESTINATION_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Conexión Segura con Outlook
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Encriptación TLS necesaria para Outlook
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, DESTINATION_EMAIL, text)
        server.quit()
        
        print(f"📧 Email enviado vía Outlook a {DESTINATION_EMAIL}")
        return True
    except Exception as e:
        print(f"❌ Error enviando email (Outlook): {e}")
        return False

# --- PRUEBA RÁPIDA ---
if __name__ == "__main__":
    print("Enviando email de prueba...")
    send_alert("TEST", "PRUEBA", 100.00, 0.99, "Probando conexión Outlook")