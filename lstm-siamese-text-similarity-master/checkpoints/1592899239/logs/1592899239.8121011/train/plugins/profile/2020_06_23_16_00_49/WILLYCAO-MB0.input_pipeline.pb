	j�t�x5@j�t�x5@!j�t�x5@	 L&�L	@ L&�L	@! L&�L	@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	j�t�x5@�~j�t��?A�� �rh4@Y�|?5^��?*	     @�@2F
Iterator::Model�(\����?!ҏ~���V@)�~j�t��?1��xƓV@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�&1��?!jS��Ԧ@)�&1��?1jS��Ԧ@:Preprocessing2S
Iterator::Model::ParallelMap�~j�t��?!�{��^��?)�~j�t��?1�{��^��?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����Mbp?! ��G?��?)����Mbp?1 ��G?��?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����Mb`?! ��G?��?)����Mb`?1 ��G?��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�~j�t��?�~j�t��?!�~j�t��?      ��!       "      ��!       *      ��!       2	�� �rh4@�� �rh4@!�� �rh4@:      ��!       B      ��!       J	�|?5^��?�|?5^��?!�|?5^��?R      ��!       Z	�|?5^��?�|?5^��?!�|?5^��?JCPU_ONLY