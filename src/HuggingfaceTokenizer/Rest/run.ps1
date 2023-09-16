$env:DOCKER_SCAN_SUGGEST = "false"
$imageId = docker build -q .
docker run --rm -it -p 8080:5000 -v "${env:HUGGINGFACE_TOKENIZER_CONFIG}:/tokenizer" $imageId
