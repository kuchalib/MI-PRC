$log = "log_zebra1.csv"
$hashes = "d85fb95cb761f5874f35ce32c305739b"
$alphabets = 4
$originalValues = "zebra1"
$min = 6
$max = 6
$THREADS = 64, 96, 128
$BLOCKS = 16,32,48,64,80,96,112,128,144,160
#$BLOCKS = 200,220,240,260,280,300,320,340,360,380,400 
"sep=;" >> $log
"Pass;Blocks;Threads;Time" >> $log
$alphabet = $alphabets
$hash = $hashes
$minA = $min
$maxA = $max
foreach ($THREAD in $THREADS)
{
    foreach ($BLOCK in $BLOCKS)
    {
        $i = Get-Date
        $ret = .\prc_gpu.exe 2 $alphabet $hash $minA $maxA $BLOCK $THREAD
        $j = Get-Date
        #($j - $i).TotalMilliseconds
        $ret + ";" + $BLOCK + ";" + $THREAD + ";" + ($j - $i).TotalMilliseconds >> $log  
    }
}