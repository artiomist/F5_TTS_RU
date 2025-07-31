# run_cli.py
'''
python .\run_f5_app.py --gpu 1 --port 7861
python .\run_f5_app.py --gpu 0 --port 7860
'''
import sys
import click
from f5_tts_ru_utils import config
import logging
logging.basicConfig(
    level=config.LOGGINGLEVEL,
    format=config.LOGGINGFORMAT, 
)
from f5_tts_ru_app import create_gradio_app

@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Port to run the app on")
@click.option("--host", "-H", default="0.0.0.0", help="Host to run the app on")
@click.option("--share", "-s", is_flag=True, default=False, help="Share via Gradio link")
@click.option("--api", "-a", is_flag=True, default=True, help="Allow API access")
@click.option("--gpu", type=int, default=config.CURRENT_GPU, help="Which GPU to use (e.g. 0 or 1)")
def main(port, host, share, api, gpu):
    logging.info(f"🔧 Using GPU {gpu} on port {port} | Split Inference between 2 GPU: {config.SPLIT_INFERENCE_BETWEEN_TWO_GPU}")
    config.CURRENT_GPU = gpu

    app = create_gradio_app()
    app.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        debug=True
    )

if __name__ == "__main__":
    try:
        import spaces
        USING_SPACES = True
        logging.info("Running inside HuggingFace Spaces.")
    except ImportError:
        USING_SPACES = False
        logging.info("Running locally.")

    if not USING_SPACES:
        logging.debug("Starting CLI entry point...")
        logging.debug(f"CLI args: {sys.argv}")
        try:
            main()
        except Exception as e:
            logging.error(f"❌ Error: {e}", exc_info=True)
            sys.exit(1)
    else:
        app = create_gradio_app()
        logging.info("🚀 Launching Gradio app...")
        app.queue()
        app.launch(inline=False, share=True)  # important for letting logging work after this line
        logging.info("✅ Gradio interface has been successfully loaded.")
