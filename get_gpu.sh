# cmd0=`nvidia-smi | tail -n +15 | grep " 0 "`
# cmd1=`nvidia-smi | tail -n +15 | grep " 1 "`

doing=0

while [ $doing != 1 ]; do
	cmd0=`nvidia-smi | tail -n +15 | grep " 3011 "`
	cmd1=`nvidia-smi | tail -n +15 | grep " 15920 "`

	if [ ! "$cmd0" ]; then
		doing=1
		echo "0 is free"
		ipython ./2.py 0
	elif [ ! "$cmd1" ]; then
		doing=1
		echo "1 is free"
		ipython ./2.py 1
	fi

	if [ $doing != 1 ]; then
		echo "no free gpu"
		sleep 10
	fi
done
