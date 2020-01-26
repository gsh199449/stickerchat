Dataset for WWW 2020 paper "Learning to Respond with Stickers: A Framework of Unifying Multi-Modality in Multi-Turn Dialog"

# How to get the dataset
Signed the following copyright announcement with your name and organization. Then complete the form online(https://forms.gle/APosP3W9vganmx9G6) and mail to shengao#pku.edu.cn ('#'->'@'), we will send you the corpus by e-mail when approved.

# Copyright
The original copyright of all the conversations belongs to the source owner.
The copyright of annotation belongs to our group, and they are free to the public.
The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes and distributed to others.

# What's in the dataset

`npy_stickers`: is a directory which stores all the sticker set. Each sticker is convered into a numpy matrix and stored in an `.npy` file.

`inception_v3*`: are the checkpoint files which stores the parameter of the Inception_V3 model. The parameters in this checkpoint is copied from the checkpoint released by TensorFlow. But the original checkpoint file released by TensorFlow can not be used at this project, since we change the name scope of the variables to load in our model.

`release_test.json`: test dataset, contains 10000 lines.

`release_train.json`: train dataset, contains 320168 lines.

`release_val.json`: validation dataset, contains 10000 lines.

`vocab`: message vocabulary, each line contains one word.

`emoji_vocab`: vocabulary of emojis, each line contains emoji.

# Data Sample
Each line in the JSON file represents one data sample, which contains the dialog history and the ground truth sticker.

|  Json Key Name  | Description                                |
|:---------------:|--------------------------------------------|
| reply_to_msg_id | current message is a reply for message-id  |
| id              | message-id                                 |
| from_id         | user-id                                    |
| sticker_set_id  | id of sticker set                          |
| sticker_id      | sticker id                                 |
| sticker_alt     | emoji label of this sticker                |
| text            | each word is split by the whitespace       |

```json
{
    "context": [
        {
            "text": "\u6211 \u6b63\u5728 \u7528 \u5927 \u5957\u5b50 \uff0c \u597d\u4e45 \u6ca1\u7528 \u4e86",
            "reply_to_msg_id": null,
            "id": 52000,
            "from_id": 284336329
        },
        {
            "text": "\u6ca1 \u60f3\u5230 \u8fd8 \u80fd \u4e0a TG",
            "reply_to_msg_id": null,
            "id": 52002,
            "from_id": 284336329
        },
    ],
    "current": {
        "from_id": 271484871,
        "reply_to_msg_id": null,
        "sticker_set_id": 556029562811580420,
        "sticker_id": 556029562811581404,
        "sticker_alt": "\ud83d\ude00"
    }
}
```

Note that the `sticker_alt` is given by the author of the sticker set, some of the authors use random emoji labels in their sticker set.

# Citation

```bibtex
@inproceedings{gao2020sticker,
  title={Learning to Respond with Stickers: A Framework of Unifying Multi-Modality in Multi-Turn Dialog},
  author={Gao, Shen and Chen, Xiuying and Li, Mingzhe and Liu, Li and Zhao, Dongyan and Yan, Rui},
  booktitle = {The World Wide Web Conference (WWW '20)},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  year = {2020}
}
```



