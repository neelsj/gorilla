REMOTE_OPENAI_BASE_URL=http://localhost:10000/v1 \
REMOTE_OPENAI_TOKENIZER_PATH=/datadisk/checkpoints/neel-p0-phi4mm-4b-triplemath-0202-r-notv2-qfb98/checkpoints/checkpoint-27276/ \
python -m bfcl_eval generate --model microsoft/Phi-4-reasoning-vision-5B-base --skip-server-setup --include-input-log --test-category non_live

#REMOTE_OPENAI_BASE_URL=http://localhost:10001/v1 \
#REMOTE_OPENAI_TOKENIZER_PATH=/datadisk/checkpoints/neel-p0-phi4mm-4b-multi-16k-0321-r-rai-tool-g9dkt/checkpoints/checkpoint-4274/ \
#python -m bfcl_eval generate --model microsoft/Phi-4-reasoning-vision-5B --skip-server-setup --include-input-log --test-category non_live
