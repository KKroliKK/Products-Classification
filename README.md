# Products-Classification

I solved the [**test task from KazanExpress**](https://remarkable-cockatoo-98b.notion.site/KazanExpress-d1c9915724544e1693afd0ca1b61a957) on multiclass products classification. Each product has its _title_, _description_, _characteristics_, _images_ and other data that is usually included into product cards on the marketplace. I developed a `multimodal model` which uses embeddings from `ruBERT`, `FastText`, `Wiki2Vec` and feature maps obtained from `ResNet` pretrained on the ImageNet. My solution performs 0.884 on f1 weighted score on more than 750 classes.
