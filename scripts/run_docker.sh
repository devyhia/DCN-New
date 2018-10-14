RUNTIME=''

for i in "$@"
do
case $i in
    --gpu)
        RUNTIME='--runtime=nvidia'
    shift
    ;;
esac
done

docker run -v `pwd`:/usr/local/src/code $RUNTIME --rm -it dcn-new bash
