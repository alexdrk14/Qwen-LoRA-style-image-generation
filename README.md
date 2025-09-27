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

Just provide pairs of images and text description as described in [repo](https://github.com/FlyMyAI/flymyai-lora-trainer) with use of RTX 4090 training configuration. This configuration will store checkpoints that could be used for the further inference of the model. 

```bash
#update the img_dir: ./your_lora_dataset with your directory of the image-text pairs in the: train_configs/train_lora_4090.yaml 
accelerate launch train.py --config ./train_configs/train_lora.yaml

#this will store the checkpoints in the ./test_lora_saves folder 
#After you could run the inference of the model.

uv run inference.py --model_name 'Qwen/Qwen-Image' --lora_weights "Path to the LoRA weights." --output_image 'test_image.jpg' --prompt "Your prompt description for image" --width 1024 --height 1024 --num_inference_steps 50

#This will store result in the test_image.jpg file.
```


## Execution: 

```bash
#Sync libs 
uv sync
#Execute the main script that do the pipeline
uv run main.py

```

For the model tuning just use the dataset from processed_dataset folder with the https://github.com/FlyMyAI/flymyai-lora-trainer.git repository it is very straight forward usage.