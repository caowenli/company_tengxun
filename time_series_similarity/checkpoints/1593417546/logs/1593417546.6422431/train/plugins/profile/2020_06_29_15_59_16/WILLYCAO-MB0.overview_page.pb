�	�G�zS�@�G�zS�@!�G�zS�@	+�f���?+�f���?!+�f���?"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'��G�zS�@B`��"۹?A+��Q�@Y��|?5^�?*	     �e@2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapsh��|?�?!���� H@)sh��|?�?1���� H@:Preprocessing2F
Iterator::Model�MbX9�?!qG��F@)�V-�?1�;⎸�D@:Preprocessing2S
Iterator::Model::ParallelMap����Mb�?!��)kʚ@)����Mb�?1��)kʚ@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat����Mb�?!��)kʚ@)����Mb�?1��)kʚ@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����MbP?!��)kʚ�?)����MbP?1��)kʚ�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor����MbP?!��)kʚ�?)����MbP?1��)kʚ�?:Preprocessing2R
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	B`��"۹?B`��"۹?!B`��"۹?      ��!       "      ��!       *      ��!       2	+��Q�@+��Q�@!+��Q�@:      ��!       B      ��!       J	��|?5^�?��|?5^�?!��|?5^�?R      ��!       Z	��|?5^�?��|?5^�?!��|?5^�?JCPU_ONLY2black"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 