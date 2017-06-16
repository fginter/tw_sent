import six
assert six.PY3, "Run me with python3"
import csv
import random
import math

def read_comments(fname):
    """Reads the "komentit.csv" file"""
    examples={} #class -> list of examples
    
    rows=[]
    with open(fname) as f:
        for line in f:
            line=line.rstrip("\n")
            cols=line.split("\t")
            rows.append(cols)

    labels=rows[0] #labels is first row
    rows=rows[1:]  #rest is data
    for col_idx,label in enumerate(labels):
        label=label.lower()
        if label.endswith("+") or label.endswith("-"):
            label=label[:-1]+" "+label[-1] #sometimes the + is attached to the label
            prop,plus_minus=label.split()
            prop_path="/"+prop+"/"+prop+plus_minus # /laadukas/laadukas+
        else:
            prop_path="/"+label
        for r in rows:
            example=r[col_idx].strip()
            if not example: #empty row in this column
                continue
            examples.setdefault(prop_path,[]).append(example)
    return examples

def print_examples(example_dict,train_file,test_file,test_proportion):
    for prop_path, example_list in example_dict.items():
        random.shuffle(example_list)
        examples=list(('"'+example.replace('"',"'")+'"',prop_path) for example in example_list)
        test=math.ceil(len(example_list)*test_proportion)
        for e,p in examples[:test]:
            print(e,p,file=test_file,sep=",")
        for e,p in examples[test:]:
            print(e,p,file=train_file,sep=",")
        

examples=read_comments("data/komentit.csv")
keywords=read_comments("data/sanasto.csv")
with open("komentit.train.csv","w") as train_f,open("komentit.test.csv","w") as test_f:
    print_examples(examples,train_f,test_f,0.2)
    print_examples(keywords,train_f,None,0.0) #add each keyword as example
    
    

