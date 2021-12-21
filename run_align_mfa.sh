# LEXICON_PATH="aligner/montreal-forced-aligner/pretrained_models/librispeech-lexicon.txt"
# LANGUAGE="english"
# DATA_PATH="corpus/LJSpeech-1.1/wavs"
# PREPROCESSED_PATH="preprocessed/LJSpeech/TextGrid"

# ./aligner/montreal-forced-aligner/bin/mfa_align $DATA_PATH $LEXICON_PATH $LANGUAGE $PREPROCESSED_PATH -j 8

base_path="corpus/NgocHuyen/NgocHuyen/data_modeling_word"
mfa train \
    $base_path/lab_ngochuyen_word \
    $base_path/lexicon.txt \
    $base_path/TextGrid \
    -o "/home/nguyenlm/Documents/MFA/pretrained_models/acoustic/NgocHuyen" \
    -j 8 \
    --clean

### adaptation ###
base_path="corpus/VLSP2021/VLSP2021/data_modeling_word"
mfa adapt \
    $base_path/lab_vlsp_word \
    $base_path/lexicon.txt \
    maiphuong \
    /home/nguyenlm/Documents/MFA/lab_vlsp_word/acoustic_model \
    -j 8 \
    --clean

### align with pretrained model ###
base_path="corpus/VLSP2021/VLSP2021/data_modeling_word"
mfa align \
    $base_path/lab_vlsp_align_word \
    $base_path/lexicon.txt \
    /home/nguyenlm/Documents/MFA/lab_vlsp_word/acoustic_model.zip \
    $base_path/TextGrid \
    --clean
