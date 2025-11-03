import uvicorn

from app import create_app
from app.config import parse_args


def main():
    args = parse_args()
    application = create_app(args)
    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
