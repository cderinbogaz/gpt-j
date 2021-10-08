pip3 install torch

git clone https://github.com/kingoflolz/mesh-transformer-jax.git
pip3 install -r mesh-transformer-jax/requirements.txt

pip3 install mesh-transformer-jax/ jax==0.2.12 jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip3 install git+https://github.com/finetuneanon/transformers@gpt-j

sudo apt install zstd

time wget -c https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd

time tar -I zstd -xf step_383500_slim.tar.zstd

mkdir gpt-j-hf
curl https://gist.githubusercontent.com/finetuneanon/a55bdb3f5881e361faef0e96e1d41f09/raw/e5a38dad34ff42bbad188afd5e4fdb2ab2eacb6d/gpt-j-6b.json > gpt-j-hf/config.json
