### GitHub issue assignment

This repo contains the pipeline to assign github issues to their respective labels through fine-tuned BERT model. The model is pushed at: https://huggingface.co/SarmadBashir/Issue_Assignment. The developed pipeline will download the model once, in order to make the predictions.

### How to run?

1. Clone the repo
2. Create the docker image: `docker image build -t vaadin .`
3. Run the image as container: `docker run -p 5001:5000 -d vaadin`
4. Check if the container is up and running: `docker ps`

### Invoke the Flask API running inside docker container

``` bash
curl --location --request GET "http://127.0.0.1:5001/assigner" \
--header "Content-Type: application/json" \
--data-raw "{\"text\": \"improves compatibility with java and java there might also be other potentially important improvements in other versions\"}"
```
### Output
```
{
    "prediction": {
        "label": "enhancement",
        "probability": 0.92
    }
}
```
### About Model

After preprocessing the provided datasets, following labels are trained and will be predicted by the pipeline:
| Labels               | No. of training rows | 
| -------------        |:-------------:       |
| bug                  | 1724        |
| enhancement          | 1304             |   
| internal improvement | 212             |
| hilla                | 197             |
| a11y                 | 179             |
| documentation        | 76             |
