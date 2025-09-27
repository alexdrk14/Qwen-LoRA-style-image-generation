# Qwen-LoRA-style-image-generation

This project is created in order to allow Qwen model to procude more accurate results based on the speccific artist design structure. 

Steps: 
- Downloading image from artist URL (in our case: https://www.simonstalenhag.se with total of 110 images)
- Labeling images with text description with use of Salesforce/instructblip-vicuna-7b Model. 
- Lora fine-tuning is based on the repo: https://github.com/FlyMyAI/flymyai-lora-trainer.git



## Data collection

Data crawling are made with beatifulsoup and requests library. All images are stored in ./images folder, and in case when images are already downloaded to folder script wit skip the download procedure. Crawler is designed to collect all categories of the sketchs provided by the designer in the web service: https://www.simonstalenhag.se .

## Image standarization and labeling 

Images are resized into the 5 categories according to the aspect ration of the original image ("square", "portrait", "landscape", "wide" and "tall"). After identification more apropriate image ratio category image will be resized to the lower resolution. After the image are resized it stored separetely in the ./processed_dataset folder, after the labeling procedure is complited ./processed_dataset/train_metadata.json file is created and contain description for each image.

## LoRA fine-tuning 

TODO 

## Evaluation 

TODO

## Execution: 

```bash
#Sync libs 
uv sync
#Execute the main script that do the pipeline
uv run main.py

```

