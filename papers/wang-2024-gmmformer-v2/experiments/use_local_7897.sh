# Source on 5090_1 only. Laptop clash is 127.0.0.1:7897; this host uses reverse tunnel 17897.
export http_proxy=http://127.0.0.1:17897
export https_proxy=http://127.0.0.1:17897
export all_proxy=http://127.0.0.1:17897
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export ALL_PROXY="$all_proxy"
export no_proxy=localhost,127.0.0.1
export NO_PROXY="$no_proxy"
