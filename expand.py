import sys
import re
import lwvlib

commaRe=re.compile(", ?")

class Sentiment:

    def __init__(self,major,minor,words):
        self.major=major
        self.minor=minor
        self.orig=set(words)
        self.expanded=[]

    def expand(self,wv,by=100):
        all_expansions=[] #(score,word)
        for candidate in {self.minor}|self.orig:
            nn_list=wv.nearest(candidate,100)
            if not nn_list:
                continue
            for score,nn in nn_list:
                if nn not in self.orig:
                    all_expansions.append((score,nn))

        exp=set()
        all_expansions=sorted(all_expansions,reverse=True)
        counter=0
        for score,nn in all_expansions:
            if nn not in exp:
                self.expanded.append(nn)
                exp.add(nn)
                counter+=1
                if counter==by:
                    break
    
    def __str__(self):
        return "## {}\n\n  Orig\n: {}\n   Expanded\n: {}".format(self.minor,", ".join(sorted(self.orig)),", ".join(sorted(self.expanded)))


def get_sents(inp):
    major=None
    minor=None
    ordered=[] #(major,minor)
    sents={} #(major,minor) -> Sentiment()
    for line in inp:
        line=line.strip().lower()
        if not line:
            continue
        if line.startswith("## "):
            minor=line.split()[1]
            ordered.append((major,minor))
        elif line.startswith("# "):
            major=line.split()[1]
        else:
            words=set(commaRe.split(line))
            sents[(major,minor)]=Sentiment(major,minor,words)
    return ordered,sents



if __name__=="__main__":
    wv=lwvlib.load("../w2v/pb34_wf_200_v2_skgram.bin",40000,100000)
    ordered,sents=get_sents(sys.stdin)
    for s in sents.values():
        s.expand(wv,50)
    m=None
    for (major,minor) in ordered:
        if major!=m:
            print("#",major)
            print()
            m=major
        print(sents[(major,minor)])
        print()

