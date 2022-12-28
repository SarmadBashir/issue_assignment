### GitHub issue assignment

This repo contains the pipeline to assign github issues to their respective labels through fine-tuned BERT model. The model is pushed at: https://huggingface.co/SarmadBashir/Issue_Assignment. The pipeline will download and make the predictions from the model.

### How to run?

1. Clone the repo
2. Create the docker image: `docker image build -t vaadin .`
3. Run the image as container: `docker run -p 5001:5000 -d vaadin`
4. Check if the container is up and running: `docker ps`

### Invoke the flask api running inside docker container

``` bash

```
