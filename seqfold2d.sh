#!/usr/bin/env bash

set -e -o posix

declare -a saved_models=(
    "bprna.TR0VL0.2p2M"
    "bprna.TR0VL0.3p5M"
    "bprna.TR0VL0.960K"
    "stralign.ND.1p4M"
    "stralign.ND.1p4M.alpha300"
    "stralign.ND.960K"
    "stralign.NR80.1p4M"
    "stralign.TRVL.960K"
    "strive.tRNA.400K"
)
default_model=${saved_models[2]}

[[ ${#@} -lt 1 ]] && {
    echo -ne "
Usage: ${0##*/} action data_files [cmOptions]

Arguments:
    action          : one of train, evaluate/eval, predict
    data_files      : one or multiple files of fmt: fasta or csv or pkl
    -model       [] : choose a model from the following (default: ${default_model}):
"
    printf "%47s\n" "${saved_models[@]}"
    # -devset      [] : one of stral-ND, stral-NR80, bprna-TR0VL0 (case insensitive)
    # -params      [] : one of 400K, 960K, 1.4M, 3.5M (case insensitive)
echo -e "
    -cmdArgs        : all other options are passed to fly_paddle.py as-is

    SPACE in folder/file names will very likely BREAK the code!!!
"
    exit 0
}

start_time=`date +%s`
BASH_HIST="$(pwd)/.bash_history"
echo "=======>>> $(date +'%Y-%m-%d %H:%M:%S') RUN-$$> ${0##*/} ${@}" >> $BASH_HIST

# get the optional and positional parameters
num_psargs=0
while read -r var ; do vars_before+=("$var") ; done <<< $(set)
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) $0 ; exit 0 ;;
        *-dryrun)
            declare ${1##*-}=TRUE ; shift
            ;;
        *-model|*-devset|*-params)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                declare ${1##*-}="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        -*) # unrecognized options will be saved to cmdArgs
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                [[ -z "${cmdArgs}" ]] && cmdArgs="$1 $2" || cmdArgs="${cmdArgs} $1 $2"
                shift 2
            else
                [[ -z "${cmdArgs}" ]] && cmdArgs="$1" || cmdArgs="${cmdArgs} $1"
                shift
            fi
            ;;
        *) # positional arguments
            num_psargs=$((num_psargs + 1))
            case ${num_psargs} in
                1) action="$1" ;;
                *) [[ -z "${data_files}" ]] && data_files="$1" || data_files="${data_files} $1" ;;
            esac
            shift
            ;;
    esac
done

# default values
# [[ -z "${devset}" ]] && devset="bprna-TR0VL0"
# [[ -z "${params}" ]] && params=960K
# devset="${devset// /}" ; devset="${devset//-/}" ; devset="${devset//_/}"
# params="${params// /}" ; params="${params//-/}" ; params="${params//_/}"

[[ -z "${model}" ]] && model=${default_model}
src_dir="$(dirname $0)/src"

case "${action^^}" in
    TRAIN|EVALUATE|EVAL|PREDICT)
        [[ -z "${data_files}" ]] && {
            echo "ERROR:: data files must be provided!!!"
            exit 1
            }
        ;;
    *)
        echo "ERROR: unsupported action: ${action}"
        exit 1
        ;;
esac

# action
load_dir="models/${model}"
cmdArgs="${cmdArgs} -feats_nn 1 -loss_fn softmax+fscore -net_summary false -verbose 0"

# show changed variables
while read -r var ; do vars_after+=("$var") ; done <<< $(set)
echo "Parameters:"
for ((i = 0; i < ${#vars_after[@]}; i++)) ; do
    var="${vars_after[$i]}"
    [[ ${var:0:1} == '[' ]] && continue # skip command history
    [[ "vars_before vars_after BASH_REMATCH PIPESTATUS" =~ "${var%%=*}" ]] && continue # skip myself
    [[ " ${vars_before[@]} " =~ " $var " ]] && continue # skip unchanged
    echo "    $var"
    echo "$var" >> $BASH_HIST
done


case "${action^^}" in
    TRAIN)
        python3 ${src_dir}/fly_paddle.py train -load_dir ${load_dir} -resume true -save_dir -data_dir ./ -data_name ${data_files} ${cmdArgs}
    ;;
    EVALUATE|EVAL)
        python3 ${src_dir}/fly_paddle.py evaluate -load_dir ${load_dir} -save_dir ./ ${data_files} ${cmdArgs}
    ;;
    PREDICT)
        python3 ${src_dir}/fly_paddle.py predict -load_dir ${load_dir} -save_dir ./ ${data_files} ${cmdArgs} -save_individual -named_after id
    ;;
    *)
        echo "ERROR: unsupported action: ${action}"
        exit 1
    ;;
    
esac

end_time=`date +%s`
run_time=$((end_time - start_time))
echo "Total running time: ${run_time} seconds (`date -d@${run_time} -u +%Hh:%Mm:%Ss`)"
echo "(^_^) @ $(date)"
echo "$(date +'%Y-%m-%d %H:%M:%S') END-$$> ${0##*/} ${@}" >> $BASH_HIST
exit 0
