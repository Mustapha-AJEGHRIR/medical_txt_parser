# File structure

```bash
medical_txt_parser
	├── Explication dataset/
	├── train_data/
		├── beth/
			├── ast/
				...
				└── record-13.ast
			├── concept
				...
				└── record-13.con
			├── rel
				...
				└── record-13.rel
			├── txt
				...
				└── record-13.txt
		└── partners/
			├── ast/
				...
				└── record-10.ast
			├── concept
				...
				└── record-10.con
			├── rel
				...
				└── record-10.rel
			├── txt
				...
				└── record-10.txt
	
	└── src/                
```


# How to use 
## Evaluation
First, we need to download the submodule for evaluation :
```bash
$ git submodule init
$ git submodule update
```