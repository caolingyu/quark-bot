docker build -t quark_bot . --no-cache
docker run -d -v ./config.py:/config/config.py --name quark_bot_container quark_bot
