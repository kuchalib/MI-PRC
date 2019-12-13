$log = "log_DICT_test.csv"
$hashes = "b4ba89b4d21e4b5886c8dae0657396de"
$originalValues = "napalm5Zq"
$dictionary = "E:\\Dictionary\words.txt"
$THREADS = 64,96,128
$BLOCKS = 16,32,48,64,80,96,112,128,144,160
#$BLOCKS = 200,220,240,260,280,300,320,340,360,380,400 
#$BLOCKS = 160
"sep=;" >> $log
"Pass;Blocks;Threads;Time" >> $log
$alphabet = $alphabets
$hash = $hashes
$minA = $min
$maxA = $max
foreach ($THREAD in $THREADS)
{
    foreach ($BLOCK in $BLOCKS)
    #for ($BLOCK  = 16; $BLOCK  -le 160; $BLOCK++)
    {
        $i = Get-Date
        $ret = .\prc_gpu.exe 3 $dictionary $hash 1 0 4 4 $BLOCK $THREAD
        $j = Get-Date
        #($j - $i).TotalMilliseconds
        $ret + ";" + $BLOCK + ";" + $THREAD + ";" + ($j - $i).TotalMilliseconds >> $log  
    }
}