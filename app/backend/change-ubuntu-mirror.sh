#!/usr/bin/env bash

SL=/etc/apt/sources.list

PARAM="r:hm:dna"

KAKAO=mirror.kakao.com
NEOWIZ=ftp.neowiz.com
HARU=ftp.harukasan.org

if [ "$(id -u)" != "0" ]; then
   echo "'$0' must be run as root" 1>&2
   exit 1
fi

function  usage {
    echo "USAGE: $0 [OPTION] ";
    echo -e "\r\n-b : make backup file";
    echo "-r[sources.list] : specify source list file (default ${SL})"
    echo "-m[mirror-url] : speficy mirror site url"
    echo "-k : use kakao mirror (${KAKAO})"
    echo "-n : use neowiz mirror (${NEOWIZ})"
    echo "-a : use harukasan mirror (${HARU})"

    exit 0;
}

REPOS=${KAKAO}

while getopts $PARAM opt; do
    case $opt in
        r)
            echo "-r option was supplied. OPTARG: $OPTARG" >&2
            SL=$OPTARG;
            ;;
        m)
            echo "Using mirror repository(${OPTARG})." >&2
            REPOS=${OPTARG}
            ;;
        k)
            echo "Using Kakao repository(${KAKAO})." >&2
            REPOS=${KAKAO}
            ;;
        n)
            echo "Using neowiz repository(${NEOWIZ})." >&2
            REPOS=${NEOWIZ}
            ;;
        a)
            echo "Using harukasan repository(${HARU})." >&2
            REPOS=${HARU}
            ;;
        h)
            usage;
             ;;
    esac
done

echo "using repository(${REPOS})"

## change mirror
sed -i.bak -re "s/([a-z]{2}.)?archive.ubuntu.com|security.ubuntu.com/${REPOS}/g" ${SL}

## check
apt update
