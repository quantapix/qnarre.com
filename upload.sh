#!/bin/bash

set -x -e -u -o pipefail


site_upload() {
    bundle exec jekyll build
    aws s3 cp _site s3://qnarre.com/ --acl public-read --recursive
}


show_usage() {
    echo "Usage: $(basename "$0") [-u]"
}

main() {
    local OPTIND=1

    while getopts "h" opt; do
	      case $opt in
	          *) show_usage; return 1;;
	      esac
    done
    shift $((OPTIND-1))

    site_upload
}

main "$@"
