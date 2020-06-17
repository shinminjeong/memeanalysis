source activate meme

pip install -e .
pip install . --upgrade

# export current directory as project path
echo
export MEME_DATA_PATH=$PWD/data
echo "MEME_DATA_PATH =" $MEME_DATA_PATH

export NLTK_DATA="/Users/minjeongshin/Work/mmkg/data/nltk"
echo "NLTK_DATA =" $NLTK_DATA

export PATH=$PATH:/usr/local/mysql/bin
