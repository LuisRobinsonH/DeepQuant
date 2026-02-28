import asyncio
from telegram import Bot

TELEGRAM_TOKEN = '8595554468:AAF_a9CR9zj2_352MLy6p_dHkfq20pKE_Xg'
TELEGRAM_CHAT_ID = 6351372403

def send_telegram_alert(message: str):
    async def _send():
        bot = Bot(token=TELEGRAM_TOKEN)
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            print('Mensaje enviado a Telegram.')
        except Exception as e:
            print('Error al enviar mensaje a Telegram:', e)
    asyncio.run(_send())
