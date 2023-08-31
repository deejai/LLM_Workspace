found here:
https://github.com/facebookresearch/llama/issues/540


```
Skip to content
facebookresearch
/
llama

Type / to search

Code
Issues
442
Pull requests
81
Actions
Projects
Security
Insights
4 Bit Inference of LLaMA-2-70B #540
Closed
ijoffe opened this issue on Jul 25 · 16 comments
Closed
4 Bit Inference of LLaMA-2-70B
#540
ijoffe opened this issue on Jul 25 · 16 comments
Comments
@ijoffe
ijoffe commented on Jul 25
Has anyone been able to get the LLaMA-2 70B model to run inference in 4-bit quantization using HuggingFace? Here are some variations of code that I've tried based on various guides:

name = "meta-llama/Llama-2-70b-chat-hf"    # I've also tried vanilla "meta-llama/Llama-2-70b-hf"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.float16,
    load_in_4bit=True,    # changing this to load_in_8bit=True works on smaller models
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)

generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)
name = "meta-llama/Llama-2-70b-chat-hf"    # I've also tried vanilla "meta-llama/Llama-2-70b-hf"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",    # I've also tried removing this line
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,    # I've also tried removing this line
)
model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False    # I've also tried removing this line
model.gradient_checkpointing_enable()    # I've also tried removing this line
model = prepare_model_for_kbit_training(model)    # I've also tried removing this line

generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)
When running all of these variations, I am able to load the model on a 48GB GPU, but making the following call produces an error:

text = "any text"
response = generation_pipe(
    text,
    max_length=128,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
The error message is as follows:

RuntimeError: shape '[1, 410, 64, 128]' is invalid for input of size 419840
What am I doing wrong? Is this even possible? Has anyone been able to get this 4-bit quantization working?

@sfingali
sfingali commented on Jul 26
Having the same issue.

@marcelbischoff
marcelbischoff commented on Jul 26
I remember fixing this error by updating transformers to 4.31

@sfingali
sfingali commented on Jul 26
I'm on 4.31.
I actually had this working a few days ago but now it's not.

@ijoffe
Author
ijoffe commented on Jul 26
Thanks for the replies! Apparently, one of my virtual environments was still on transformers 4.30.0, so upgrading to 4.31.0 fixed the issue.

For anyone else experiencing this, these were the package versions that solved the problem for me (from running pip list):

Package                  Version
------------------------ ----------
accelerate               0.21.0
bitsandbytes             0.41.0
certifi                  2023.7.22
charset-normalizer       3.2.0
cmake                    3.27.0
filelock                 3.12.2
fsspec                   2023.6.0
huggingface-hub          0.16.4
idna                     3.4
Jinja2                   3.1.2
lit                      16.0.6
MarkupSafe               2.1.3
mpmath                   1.3.0
mypy-extensions          1.0.0
networkx                 3.1
numpy                    1.25.1
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-cupti-cu11   11.7.101
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.2.10.91
nvidia-cusolver-cu11     11.4.0.1
nvidia-cusparse-cu11     11.7.4.91
nvidia-nccl-cu11         2.14.3
nvidia-nvtx-cu11         11.7.91
packaging                23.1
pip                      22.3.1
psutil                   5.9.5
pyre-extensions          0.0.29
PyYAML                   6.0.1
regex                    2023.6.3
requests                 2.31.0
safetensors              0.3.1
scipy                    1.11.1
setuptools               65.5.0
sympy                    1.12
tokenizers               0.13.3
torch                    2.0.1
tqdm                     4.65.0
transformers             4.31.0
triton                   2.0.0
typing_extensions        4.7.1
typing-inspect           0.9.0
urllib3                  2.0.4
wheel                    0.41.0
xformers                 0.0.20
Thanks!

@Mega4alik
Mega4alik commented on Jul 27
@ijoffe As I understand the "chat" model is built to have chat data as an input.
jSimilar to ChatGPT

[ 
{"role":"system", "<content>"},
{"role":"user", "<content>"},
{"role":"assistant", "<content>"},
]
Have you figured out how to feed such type of data into the huggingface llama2 model?

Thanks!

@m4dc4p
m4dc4p commented on Jul 27 • 
@ijoffe Could you put up gist or paste in the script you ended up with to load the 4-bit models? I can probably piece it together from your original post but a complete example would be super helpful!

@ijoffe
Author
ijoffe commented on Jul 31
For sure! This code worked for me, here it is:

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch


name = "meta-llama/Llama-2-70b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)

text = "any text "    # prompt goes here

sequences = generation_pipe(
    text,
    max_length=128,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_k=10,
    temperature=0.4,
    top_p=0.9
)

print(sequences[0]["generated_text"])
@ijoffe ijoffe closed this as completed on Jul 31
@Gavingx
Gavingx commented last month
can i run Llama-2-70b-chat-hf with 4 * RTX 3090? , is there any document i can refer to?

@ijoffe
Author
ijoffe commented last month
Not sure about any reference document, but those are 24GB GPUs right? I got this running on one 48GB GPU, so even with the parallelization overhead I bet you could get this running if you have 4.

@Gavingx
Gavingx commented last month
ok, i'll try it, Thanks a lot!

@yanxiyue
Contributor
yanxiyue commented last month • 
Not sure about any reference document, but those are 24GB GPUs right? I got this running on one 48GB GPU, so even with the parallelization overhead I bet you could get this running if you have 4.

@ijoffe
Did you run this with quantization or without? If you did use quantization, how many bits did you use?

@malaka3000
malaka3000 commented last month
@yanxiyue take a look at @ijoffe code snippet above. load_in_4bit=True is set in their quantization_config

@yanxiyue
Contributor
yanxiyue commented last month
@yanxiyue take a look at @ijoffe code snippet above. load_in_4bit=True is set in their quantization_config

thanks for the additional context!

@hassanzadeh
hassanzadeh commented 2 weeks ago
For sure! This code worked for me, here it is:

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch


name = "meta-llama/Llama-2-70b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)

text = "any text "    # prompt goes here

sequences = generation_pipe(
    text,
    max_length=128,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_k=10,
    temperature=0.4,
    top_p=0.9
)

print(sequences[0]["generated_text"])
Hey @ijoffe,
What is the exact purpose for passing the pad_token_id and eos_token_id?
THanks

@ijoffe
Author
ijoffe commented 2 weeks ago
Hey @hassanzadeh, this just ensures the tokenizer and model are on the same page hen it comes to the special tokens. I'm not sure if it's required, but it theoretically ensures the LLM stops generating output once the EOS token is reached.

@hassanzadeh
hassanzadeh commented 2 weeks ago
Hey @hassanzadeh, this just ensures the tokenizer and model are on the same page hen it comes to the special tokens. I'm not sure if it's required, but it theoretically ensures the LLM stops generating output once the EOS token is reached.

I see, thanks for your quick response :)

@deejai

 
Leave a comment
No file chosen
Attach files by dragging & dropping, selecting or pasting them.
Remember, contributions to this repository should follow its contributing guidelines, security policy, and code of conduct.
Assignees
No one assigned
Labels
None yet
Projects
None yet
Milestone
No milestone
Development
No branches or pull requests

Notifications
Customize
You’re not receiving notifications from this thread.
9 participants
@m4dc4p
@Mega4alik
@yanxiyue
@hassanzadeh
@Gavingx
@marcelbischoff
@sfingali
@ijoffe
@malaka3000
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About

```
