from image_crawler import download_images, generate_labels


def main():
    print("Hello from qwen-lora-style-image-generation!")
    print('Lets start with downloading images from Simon Stalenhag website...')
    download_images()
    print('Start labeling images using Ollama...')
    generate_labels()
    print('All done! Check out images/ folder for results.')

    print('Now follow readme to fine-tune Qwen-VL model!')
    


if __name__ == "__main__":
    main()
