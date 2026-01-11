import os
import uvicorn

from app import config as app_config
from app.config import parse_args


def main():
    args = parse_args()
    if getattr(args, "database_url", None):
        os.environ["DATABASE_URL"] = args.database_url
        app_config.DATABASE_URL = args.database_url
    from app import create_app

    application = create_app(args)
    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
