
# How to use 
## Evaluation
First, we need to download the submodule for evaluation :
```bash
$ git submodule init
$ git submodule update
```

## Build dataset
First we need to have the initial data as follows :

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
			└── txt
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
			└── txt
				...
				└── record-10.txt
	
	└── src/                
```

Then execute the following command to build the dataset from the root of the project:

```bash
$ ./src/data_merger.sh
```

