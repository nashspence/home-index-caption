# Home Index Caption

Home Index Caption is a module for [Home Index](https://github.com/nashspence/home-index). It exposes the [Home Index RPC module](https://github.com/nashspence/home-index) interface and generates captions for images and videos using the BLIP model from Hugging Face.

## Quick start

The included `docker-compose.yml` starts Home Index, Meilisearch and this caption module. After installing Docker, run:

```bash
docker compose up
```

Files are stored under `bind-mounts/` by default. Edit the compose file if you need to change any paths or environment variables.

## Running manually

To run the module outside of Docker:

```bash
pip install -r requirements.txt
python -m home_index_caption.main
```

Set the `MODULES` environment variable on your Home Index instance to the module's endpoint, e.g. `http://localhost:9000`.

Environment variables like `DEVICE`, `RESIZE_MAX_DIMENSION` and `VIDEO_NUMBER_OF_FRAMES` can be adjusted to control the caption model.
