import asyncio

from dotenv import load_dotenv
import logging
import sys
import app

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)
load_dotenv()


if __name__ == "__main__":
    asyncio.run(app.run())