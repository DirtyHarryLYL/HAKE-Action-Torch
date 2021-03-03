import time
import numpy as np
import os.path as osp
import torch
HO_weight   = np.array(
                [39.86157711904592, 42.40444304026247, 44.834823527125415, 39.73991804849558, 39.350051894572104, 43.56609116327042, 37.78352039671705, 37.27879696661561, 56.2961038839078, 27.187192997462514, 48.69942543701149, 29.31727479695975, 40.91791293317506, 41.48885009402292, 48.00306615559755, 50.012214583404685, 41.67212390491823, 42.921511271001236, 28.880648716145704, 29.52003435670287, 28.7948786160738, 39.76397874615436, 59.30640384054761, 16.891810699279997, 45.78457865943398, 38.21737256387448, 41.27866658762785, 55.327003753827235, 50.27550397062817, 35.892058594766205, 23.930101537298007, 42.1883315501357, 39.01256606369551, 46.40605772692243, 41.63484517972581, 51.90277694560517, 39.86157711904592, 44.3232983026516, 26.24108136535154, 33.525745002186696, 36.51886783101932, 28.541819963426093, 30.205498384606926, 45.882177032325544, 59.30640384054761, 19.277187457752916, 40.44149658882279, 40.27550397062818, 34.99276619895774, 45.24100203620805, 53.86572339704485, 50.012214583404685, 48.69942543701149, 17.603639967967627, 38.389734264590764, 54.535191293350984, 35.660893887007894, 47.692723818197855, 48.00306615559755, 43.80412030999668, 30.501126058559556, 32.35158707564564, 54.535191293350984, 62.31670379718742, 23.78093836099101, 48.00306615559755, 62.31670379718742, 36.67004315466653, 44.535191293350984, 42.58542526119043, 62.31670379718742, 48.892476988965356, 48.16697031747924, 34.600828988374865, 39.88632331032448, 14.392366637557455, 62.31670379718742, 56.2961038839078, 40.3301329276432, 43.50856787437951, 54.535191293350984, 43.86572339704485, 45.414742996902284, 57.545491249990796, 62.31670379718742, 34.76558113323671, 45.15667036083943, 41.8245235704856, 43.285803927267985, 31.29236674037406, 62.31670379718742, 13.477875959525374, 47.84512348376523, 38.73735532718288, 32.71199602184443, 27.673298950910752, 52.31670379718742, 38.425042953542096, 42.49399146679174, 55.327003753827235, 55.327003753827235, 45.15667036083943, 47.26520401398836, 45.414742996902284, 54.535191293350984, 41.27866658762785, 26.897169052605058, 55.327003753827235, 28.62825872892921, 27.253006626232384, 19.566586419407308, 42.44898645452497, 59.30640384054761, 51.52489133671117, 51.90277694560517, 37.18452779650803, 48.16697031747924, 39.86157711904592, 49.76397874615436, 49.30640384054761, 40.22155365176111, 49.30640384054761, 43.12592287342668, 50.27550397062817, 45.327003753827235, 41.244604100708735, 48.33730371046705, 57.545491249990796, 25.50972350021107, 48.69942543701149, 46.753678789514545, 33.13115849168469, 48.16697031747924, 40.824512670633624, 51.90277694560517, 62.31670379718742, 53.285803927267985, 42.97171928475174, 38.794878616073795, 30.481158460998806, 35.24100203620806, 31.561234183262115, 42.8228037307383, 42.31670379718742, 48.16697031747924, 21.951612511860127, 32.40887687915604, 43.92821288981487, 41.38248694556508, 57.545491249990796, 42.06364514453972, 51.52489133671117, 35.81362856586806, 30.962196803732283, 30.042979374291058, 30.29727316317092, 38.200506737555116, 50.55579120663061, 57.545491249990796, 19.54955885088451, 42.921511271001236, 49.529167787659134, 35.17340619973509, 38.13369088398997, 44.39278690220489, 55.327003753827235, 62.31670379718742, 40.55579120663061, 53.86572339704485, 2.660919516910389, 52.31670379718742, 52.31670379718742, 52.77427870279417, 22.931506545422504, 49.09451084984823, 42.631874311648076, 34.750342714728944, 39.88632331032448, 47.26520401398836, 59.30640384054761, 48.16697031747924, 62.31670379718742, 41.863474009320846, 41.21080669419493, 55.327003753827235, 25.14999404158607, 41.863474009320846, 41.59788372412617, 56.2961038839078, 59.30640384054761, 34.66001824959728, 37.45948953237162, 53.285803927267985, 28.543641286505434, 47.4030868588447, 57.545491249990796, 38.794878616073795, 26.440716499974968, 53.86572339704485, 49.76397874615436, 45.327003753827235, 38.83365516670581, 47.692723818197855, 48.33730371046705, 47.84512348376523, 59.30640384054761, 59.30640384054761, 27.72277891959511, 32.56698385420673, 43.285803927267985, 49.30640384054761, 50.27550397062817, 32.22644637631832, 19.113163469010704, 52.77427870279417, 37.17122627058457, 56.2961038839078, 41.67212390491823, 37.292432597343094, 39.26319010272118, 51.90277694560517, 48.16697031747924, 57.545491249990796, 25.21383732015851, 45.327003753827235, 43.12592287342668, 33.7494148933586, 62.31670379718742, 37.05331102328898, 59.30640384054761, 34.37182333059573, 24.936036649412728, 39.350051894572104, 39.32817303309036, 24.636245656163254, 49.09451084984823, 44.46340544707976, 42.31670379718742, 40.793820353356864, 59.30640384054761, 37.157965360070634, 40.44149658882279, 32.457950224103485, 51.17727027411905, 42.921511271001236, 29.26534060775103, 20.70871826180388, 41.27866658762785, 33.15743168021626, 34.81934064149681, 33.8906114010818, 19.271367693122244, 43.56609116327042, 47.00191462676487, 57.545491249990796, 59.30640384054761, 19.587154348705695, 62.31670379718742, 45.78457865943398, 45.78457865943398, 62.31670379718742, 59.30640384054761, 62.31670379718742, 34.842585718323186, 37.2516534731387, 42.23070207956824, 37.51663436761591, 43.56609116327042, 40.763343422536806, 47.545491249990796, 51.90277694560517, 43.50856787437951, 25.56809238980931, 42.23070207956824, 56.2961038839078, 48.16697031747924, 46.51886783101932, 39.37204153557148, 48.892476988965356, 62.31670379718742, 62.31670379718742, 57.545491249990796, 30.025006771796413, 39.692192899883125, 35.00888104052353, 39.57512530455062, 62.31670379718742, 38.606025174470055, 41.143990840629776, 23.637317288279576, 43.92821288981487, 45.882177032325544, 54.535191293350984, 51.90277694560517, 27.024967764570196, 40.88655579464647, 41.70972539365131, 37.375157857002996, 50.27550397062817, 50.55579120663061, 43.231853608400925, 46.876023353684666, 45.24100203620805, 62.31670379718742, 16.424023402560195, 52.31670379718742, 44.68242386155804, 41.41765268279344, 37.95507732677986, 45.882177032325544, 43.12592287342668, 62.31670379718742, 27.177871941076496, 46.08421089320842, 42.02286602033532, 59.30640384054761, 51.17727027411905, 59.30640384054761, 50.85542344040504, 43.340432884283004, 35.679694543290935, 39.46113070710968, 44.99276619895774, 34.09502300350725, 28.227523588719624, 62.31670379718742, 47.84512348376523, 50.55579120663061, 52.77427870279417, 32.15891623329701, 40.85542344040504, 41.863474009320846, 47.692723818197855, 54.535191293350984, 59.30640384054761, 25.613317385913, 42.44898645452497, 37.22467857387639, 35.8624811036965, 50.012214583404685, 38.532724787706044, 25.93880550356513, 40.19482775314784, 51.52489133671117, 48.69942543701149, 59.30640384054761, 36.15720328062341, 33.27496011434579, 45.414742996902284, 45.327003753827235, 59.30640384054761, 48.514591380071366, 43.50856787437951, 40.38545781364281, 54.535191293350984, 17.279205413234312, 40.67317523934305, 41.67212390491823, 62.31670379718742, 42.10481080648804, 38.87278106033631, 45.327003753827235, 30.602364787757338, 45.882177032325544, 57.545491249990796, 45.50429142343155, 38.932138861141375, 30.87719263294779, 45.982019241391555, 38.91226264878624, 33.71931813521595, 35.00081614532003, 41.8245235704856, 40.498267917739696, 35.8624811036965, 20.759093668408187, 39.09451084984823, 34.73515757751352, 46.2961038839078, 62.31670379718742, 34.66001824959728, 55.327003753827235, 23.89185955307172, 38.73735532718288, 46.753678789514545, 34.37879995027924, 50.85542344040504, 38.407352726153626, 25.571766624223923, 62.31670379718742, 55.327003753827235, 53.86572339704485, 37.2516534731387, 45.59572521783024, 44.834823527125415, 62.31670379718742, 33.41249360917828, 62.31670379718742, 57.545491249990796, 57.545491249990796, 46.63468655651747, 62.31670379718742, 62.31670379718742, 57.545491249990796, 53.86572339704485, 59.30640384054761, 28.45349805824696, 56.2961038839078, 51.17727027411905, 43.231853608400925, 55.327003753827235, 49.76397874615436, 48.69942543701149, 33.27496011434579, 46.753678789514545, 48.514591380071366, 62.31670379718742, 31.942438817781188, 62.31670379718742, 43.12592287342668, 46.753678789514545, 45.414742996902284, 41.38248694556508, 39.668525567092054, 44.60818368076598, 45.15667036083943, 55.327003753827235, 62.31670379718742, 26.128903552125276, 59.30640384054761, 46.51886783101932, 56.2961038839078, 48.892476988965356, 33.22114350477567, 39.621574355008256, 44.39278690220489, 57.545491249990796, 29.10279101407053, 47.545491249990796, 62.31670379718742, 57.545491249990796, 41.52489133671117, 39.37204153557148, 51.52489133671117, 31.147307331679862, 40.61408664323785, 38.389734264590764, 42.8228037307383, 35.92183890450156, 59.30640384054761, 51.52489133671117, 52.77427870279417, 33.01740819634154, 40.94949812562335, 35.93181122764105, 38.200506737555116, 36.188865229990064, 32.39558891931792, 44.68242386155804, 30.061026662792713, 48.892476988965356, 30.590674485088822, 21.711991869210635, 55.327003753827235, 44.12126444176874, 42.77427870279417, 49.09451084984823, 41.38248694556508, 47.692723818197855, 55.327003753827235, 33.95979808226316, 32.813055253426185, 32.26060934358462, 24.087181391712605, 55.327003753827235, 46.51886783101932, 44.46340544707976, 40.35770727309509, 36.814420266636475, 34.316410204746084, 34.1808939115055, 33.73735114999313, 27.336356560317153, 38.442805533800126, 42.631874311648076, 62.31670379718742, 47.26520401398836, 25.597573672771553, 44.535191293350984, 39.78817348738849, 42.72628987397649, 41.143990840629776, 38.62454522308599, 37.0147068151566, 41.863474009320846, 37.55999191394312, 51.90277694560517, 43.99161467012506, 62.31670379718742, 62.31670379718742, 40.793820353356864, 22.874345359252622, 48.892476988965356, 52.31670379718742, 62.31670379718742, 32.133860712922115, 38.68058399826598, 41.70972539365131, 38.5509342266223, 62.31670379718742, 52.31670379718742, 51.90277694560517, 47.545491249990796, 47.545491249990796, 62.31670379718742, 24.51353067578591, 38.73735532718288, 54.535191293350984, 37.347407316455275, 50.55579120663061, 57.545491249990796, 47.692723818197855, 59.30640384054761, 38.87278106033631, 40.44149658882279, 48.33730371046705, 62.31670379718742, 26.219692773393426, 45.59572521783024, 37.90761297653525, 44.834823527125415, 57.545491249990796, 29.810064602554988, 39.30640384054761, 38.46064106120429, 62.31670379718742, 43.02251454004449, 28.545463373722857, 44.75795524046251, 62.31670379718742, 43.12592287342668, 51.17727027411905, 47.545491249990796, 47.84512348376523, 33.17327222599302, 18.763338182063613, 59.30640384054761, 59.30640384054761, 62.31670379718742, 46.51886783101932, 53.86572339704485, 62.31670379718742, 57.545491249990796, 50.55579120663061, 44.834823527125415, 57.545491249990796, 55.327003753827235, 35.577283810846545, 38.661823948278425, 37.26520401398836, 62.31670379718742, 31.974431189481912, 51.17727027411905, 50.012214583404685, 49.09451084984823, 48.514591380071366, 21.94044709804023, 48.514591380071366, 40.643530449705665, 45.414742996902284, 41.902776945605176, 49.30640384054761, 37.87625583800666, 39.37204153557148, 50.85542344040504, 21.85956320777875, 31.97844685765432, 30.622898844067926, 54.535191293350984, 50.85542344040504, 52.77427870279417, 52.77427870279417, 31.27524829164734, 18.933732894860512, 50.55579120663061, 50.012214583404685, 62.31670379718742, 28.163630874931748, 47.84512348376523, 36.178285578426724, 45.59572521783024, 43.50856787437951, 62.31670379718742, 62.31670379718742, 19.859850154855067, 53.285803927267985, 59.30640384054761, 56.2961038839078, 43.50856787437951, 41.67212390491823]
            , dtype = 'float32').reshape(1,600) # HOI loss weight

# HO_weight   = np.array([
#                 9.192927, 9.778443, 10.338059, 9.164914, 9.075144, 10.045923, 8.714437, 8.59822, 12.977117, 6.2745423, 
#                 11.227917, 6.765012, 9.436157, 9.56762, 11.0675745, 11.530198, 9.609821, 9.897503, 6.664475, 6.811699, 
#                 6.644726, 9.170454, 13.670264, 3.903943, 10.556748, 8.814335, 9.519224, 12.753973, 11.590822, 8.278912, 
#                 5.5245695, 9.7286825, 8.997436, 10.699849, 9.601237, 11.965516, 9.192927, 10.220277, 6.056692, 7.734048, 
#                 8.42324, 6.586457, 6.969533, 10.579222, 13.670264, 4.4531965, 9.326459, 9.288238, 8.071842, 10.431585, 
#                 12.417501, 11.530198, 11.227917, 4.0678477, 8.854023, 12.571651, 8.225684, 10.996116, 11.0675745, 10.100731, 
#                 7.0376034, 7.463688, 12.571651, 14.363411, 5.4902234, 11.0675745, 14.363411, 8.45805, 10.269067, 9.820116, 
#                 14.363411, 11.272368, 11.105314, 7.981595, 9.198626, 3.3284247, 14.363411, 12.977117, 9.300817, 10.032678, 
#                 12.571651, 10.114916, 10.471591, 13.264799, 14.363411, 8.01953, 10.412168, 9.644913, 9.981384, 7.2197933, 
#                 14.363411, 3.1178555, 11.031207, 8.934066, 7.546675, 6.386472, 12.060826, 8.862153, 9.799063, 12.753973, 
#                 12.753973, 10.412168, 10.8976755, 10.471591, 12.571651, 9.519224, 6.207762, 12.753973, 6.60636, 6.2896967, 
#                 4.5198326, 9.7887, 13.670264, 11.878505, 11.965516, 8.576513, 11.105314, 9.192927, 11.47304, 11.367679, 
#                 9.275815, 11.367679, 9.944571, 11.590822, 10.451388, 9.511381, 11.144535, 13.264799, 5.888291, 11.227917, 
#                 10.779892, 7.643191, 11.105314, 9.414651, 11.965516, 14.363411, 12.28397, 9.909063, 8.94731, 7.0330057, 
#                 8.129001, 7.2817025, 9.874775, 9.758241, 11.105314, 5.0690055, 7.4768796, 10.129305, 9.54313, 13.264799, 
#                 9.699972, 11.878505, 8.260853, 7.1437693, 6.9321113, 6.990665, 8.8104515, 11.655361, 13.264799, 4.515912, 
#                 9.897503, 11.418972, 8.113436, 8.795067, 10.236277, 12.753973, 14.363411, 9.352776, 12.417501, 0.6271591, 
#                 12.060826, 12.060826, 12.166186, 5.2946343, 11.318889, 9.8308115, 8.016022, 9.198626, 10.8976755, 13.670264, 
#                 11.105314, 14.363411, 9.653881, 9.503599, 12.753973, 5.80546, 9.653881, 9.592727, 12.977117, 13.670264, 
#                 7.995224, 8.639826, 12.28397, 6.586876, 10.929424, 13.264799, 8.94731, 6.1026597, 12.417501, 11.47304, 
#                 10.451388, 8.95624, 10.996116, 11.144535, 11.031207, 13.670264, 13.670264, 6.397866, 7.513285, 9.981384, 
#                 11.367679, 11.590822, 7.4348736, 4.415428, 12.166186, 8.573451, 12.977117, 9.609821, 8.601359, 9.055143, 
#                 11.965516, 11.105314, 13.264799, 5.8201604, 10.451388, 9.944571, 7.7855496, 14.363411, 8.5463, 13.670264, 
#                 7.9288645, 5.7561946, 9.075144, 9.0701065, 5.6871653, 11.318889, 10.252538, 9.758241, 9.407584, 13.670264, 
#                 8.570397, 9.326459, 7.488179, 11.798462, 9.897503, 6.7530537, 4.7828183, 9.519224, 7.6492405, 8.031909, 
#                 7.8180614, 4.451856, 10.045923, 10.83705, 13.264799, 13.670264, 4.5245686, 14.363411, 10.556748, 10.556748, 
#                 14.363411, 13.670264, 14.363411, 8.037262, 8.59197, 9.738439, 8.652985, 10.045923, 9.400566, 10.9622135, 
#                 11.965516, 10.032678, 5.9017305, 9.738439, 12.977117, 11.105314, 10.725825, 9.080208, 11.272368, 14.363411, 
#                 14.363411, 13.264799, 6.9279733, 9.153925, 8.075553, 9.126969, 14.363411, 8.903826, 9.488214, 5.4571533, 
#                 10.129305, 10.579222, 12.571651, 11.965516, 6.237189, 9.428937, 9.618479, 8.620408, 11.590822, 11.655361, 
#                 9.968962, 10.8080635, 10.431585, 14.363411, 3.796231, 12.060826, 10.302968, 9.551227, 8.75394, 10.579222, 
#                 9.944571, 14.363411, 6.272396, 10.625742, 9.690582, 13.670264, 11.798462, 13.670264, 11.724354, 9.993963, 
#                 8.230013, 9.100721, 10.374427, 7.865129, 6.514087, 14.363411, 11.031207, 11.655361, 12.166186, 7.419324, 
#                 9.421769, 9.653881, 10.996116, 12.571651, 13.670264, 5.912144, 9.7887, 8.585759, 8.272101, 11.530198, 8.886948, 
#                 5.9870906, 9.269661, 11.878505, 11.227917, 13.670264, 8.339964, 7.6763024, 10.471591, 10.451388, 13.670264, 
#                 11.185357, 10.032678, 9.313555, 12.571651, 3.993144, 9.379805, 9.609821, 14.363411, 9.709451, 8.965248, 
#                 10.451388, 7.0609145, 10.579222, 13.264799, 10.49221, 8.978916, 7.124196, 10.602211, 8.9743395, 7.77862, 
#                 8.073695, 9.644913, 9.339531, 8.272101, 4.794418, 9.016304, 8.012526, 10.674532, 14.363411, 7.995224, 
#                 12.753973, 5.5157638, 8.934066, 10.779892, 7.930471, 11.724354, 8.85808, 5.9025764, 14.363411, 12.753973, 
#                 12.417501, 8.59197, 10.513264, 10.338059, 14.363411, 7.7079706, 14.363411, 13.264799, 13.264799, 10.752493, 
#                 14.363411, 14.363411, 13.264799, 12.417501, 13.670264, 6.5661197, 12.977117, 11.798462, 9.968962, 12.753973, 
#                 11.47304, 11.227917, 7.6763024, 10.779892, 11.185357, 14.363411, 7.369478, 14.363411, 9.944571, 10.779892, 
#                 10.471591, 9.54313, 9.148476, 10.285873, 10.412168, 12.753973, 14.363411, 6.0308623, 13.670264, 10.725825, 
#                 12.977117, 11.272368, 7.663911, 9.137665, 10.236277, 13.264799, 6.715625, 10.9622135, 14.363411, 13.264799, 
#                 9.575919, 9.080208, 11.878505, 7.1863923, 9.366199, 8.854023, 9.874775, 8.2857685, 13.670264, 11.878505, 
#                 12.166186, 7.616999, 9.44343, 8.288065, 8.8104515, 8.347254, 7.4738197, 10.302968, 6.936267, 11.272368, 
#                 7.058223, 5.0138307, 12.753973, 10.173757, 9.863602, 11.318889, 9.54313, 10.996116, 12.753973, 7.8339925, 
#                 7.569945, 7.4427395, 5.560738, 12.753973, 10.725825, 10.252538, 9.307165, 8.491293, 7.9161053, 7.8849015, 
#                 7.782772, 6.3088884, 8.866243, 9.8308115, 14.363411, 10.8976755, 5.908519, 10.269067, 9.176025, 9.852551, 
#                 9.488214, 8.90809, 8.537411, 9.653881, 8.662968, 11.965516, 10.143904, 14.363411, 14.363411, 9.407584, 
#                 5.281472, 11.272368, 12.060826, 14.363411, 7.4135547, 8.920994, 9.618479, 8.891141, 14.363411, 12.060826, 
#                 11.965516, 10.9622135, 10.9622135, 14.363411, 5.658909, 8.934066, 12.571651, 8.614018, 11.655361, 13.264799, 
#                 10.996116, 13.670264, 8.965248, 9.326459, 11.144535, 14.363411, 6.0517673, 10.513264, 8.7430105, 10.338059, 
#                 13.264799, 6.878481, 9.065094, 8.87035, 14.363411, 9.92076, 6.5872955, 10.32036, 14.363411, 9.944571, 
#                 11.798462, 10.9622135, 11.031207, 7.652888, 4.334878, 13.670264, 13.670264, 14.363411, 10.725825, 12.417501, 
#                 14.363411, 13.264799, 11.655361, 10.338059, 13.264799, 12.753973, 8.206432, 8.916674, 8.59509, 14.363411, 
#                 7.376845, 11.798462, 11.530198, 11.318889, 11.185357, 5.0664344, 11.185357, 9.372978, 10.471591, 9.6629305, 
#                 11.367679, 8.73579, 9.080208, 11.724354, 5.04781, 7.3777695, 7.065643, 12.571651, 11.724354, 12.166186, 
#                 12.166186, 7.215852, 4.374113, 11.655361, 11.530198, 14.363411, 6.4993753, 11.031207, 8.344818, 10.513264, 
#                 10.032678, 14.363411, 14.363411, 4.5873594, 12.28397, 13.670264, 12.977117, 10.032678, 9.609821
#             ], dtype = 'float32').reshape(1,600)

HO_weight   = torch.from_numpy(HO_weight)

vcoco_weight = np.array([3.3510249, 3.4552405, 4.0257854, 1.0, 4.088436, 
                        3.4370995, 3.85842, 4.637334, 3.5487218, 3.536237, 
                        2.5578923, 3.342811, 3.8897269, 4.70686, 3.3952892, 
                        3.9706533, 4.504736, 1.0, 1.4873443, 3.700363, 
                        4.1058283, 3.6298118, 1.0, 6.490651, 5.0808263, 
                        1.520838, 3.3888445, 1.0, 3.9899964], dtype = 'float32').reshape(1,29)
vcoco_weight = torch.from_numpy(vcoco_weight)

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.cnt = 0

    def update(self, val, k):
        self.avg = self.avg + (val - self.avg) * k / (self.cnt + k)
        self.cnt += k

    def __str__(self):
        """String representation for logging
        """
        # for stats
        return '%.4f' % self.avg

class Parser(object):

    def __init__(self, parser):
        parser = self.populate(parser)
        self.opt = parser.parse_args()

    def make_options(self):
        config = yaml.load(open(self.opt.config_path))
        dic = vars(self.opt)
        all(map( dic.pop, config))
        dic.update(config)
        return self.opt


    def populate(self, parser): 
        """ Paths """
        parser.add_argument('--data_path', default='', type=str, help='Data path where to find annotations')
        parser.add_argument('--data_name', default='', type=str, help='Dataset to run on : vrd, unrel')
        parser.add_argument('--logger_dir', default='./runs', type=str, help='Directory to write log and save models')
        parser.add_argument('--thresh_file', default=None ,type=str, help='Specify file for thresholding object detections')
        parser.add_argument('--exp_name', default='' ,type=str, help='Specify name for current experiment if no name given, would generate a random id')
        parser.add_argument('--config_path', default='configs/hico_trainval_base.yaml' ,type=str, help='Path to config file')

        """ Optimization """
        parser.add_argument('--momentum', default=0, type=float, help='Set momentum')
        parser.add_argument('--weight_decay', default=0, type=float, help='Set weight decay')
        parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use')
        parser.add_argument('--learning_rate', default=1e-3, type=float, help='Set learning rate')
        parser.add_argument('--lr_update', default=20, type=int, help='Number of iterations before decreasing learning rate')
        parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs for training')
        parser.add_argument('--margin', default=0.2, type=int, help='Set margin for ranking loss')
        parser.add_argument('--use_gpu', default=True, type=bool, help='Whether to run calculations on gpu')
        parser.add_argument('--sampler', default='priority_object', type=str, help='Sampler to use at training')
        parser.add_argument('--start_epoch', default=0, type=int, help='Epoch to start training. Default is 0.')
        parser.add_argument('--save_epoch', default=5, type=int, help='Save model every save_epoch')
        parser.add_argument('--batch_size', default=16, type=int,  help='Batch size')

        """ Inputs to load """
        parser.add_argument('--use_precompappearance', help='whether to use precomputed appearance features', action='store_true')
        parser.add_argument('--use_precompobjectscore', help='whether you use precomputed object scores', action='store_true')
        parser.add_argument('--use_image', help='whether to load image', action='store_true')
        parser.add_argument('--use_ram', help='whether to store features in RAM. Much faster', action='store_true')


        """ Networks to use in each branch """
        parser.add_argument('--net_unigram_s', default='', help='network for unigram subject branch')
        parser.add_argument('--net_unigram_o', default='', help='network for unigram object branch')
        parser.add_argument('--net_unigram_r', default='', help='network unigram predicate branch')
        parser.add_argument('--net_bigram_sr', default='', help='network bigram subject-predicate branch')
        parser.add_argument('--net_bigram_ro', default='', help='network bigram predicate-object branch')
        parser.add_argument('--net_trigram_sro', default='', help='network trigram subject-predicate-object branch')
        parser.add_argument('--net_language', default='', help='language network')
        parser.add_argument('--criterion_name', default='', help='criterion')
        parser.add_argument('--pretrained_model', default='', type=str, help='path to pre-trained visual net with independent classif')
        parser.add_argument('--mixture_keys', default='', type=str, help='keys to use in mixture e.g. s-r-o_sro_sr-r-ro')


        """ Options """
        parser.add_argument('--neg_GT', help='Using negatives candidates', action='store_true')
        parser.add_argument('--sample_negatives', default='among_batch', type=str, help='How to sample negatives when training with embeddings')
        parser.add_argument('--embed_size', default=128, type=int, help='Dimensionality of embedding before classifier')
        parser.add_argument('--d_hidden', default=1024, type=int, help='Dimensionality of hidden layer in projection to joint space')
        parser.add_argument('--num_layers', default=2, type=int, help='Number of projection layers')
        parser.add_argument('--network', default='', type=str, help='Model to use see get_model() from models.py')
        parser.add_argument('--train_split', default='train', type=str, help='train split : either train, trainminusval')
        parser.add_argument('--test_split', default='val', type=str, help='test split : either test, val')
        parser.add_argument('--use_gt', help='whether to use groundtruth objects as candidates', action='store_true')
        parser.add_argument('--add_gt', help='whether to use groundtruth objects as additional candidates during training', action='store_true')
        parser.add_argument('--use_jittering', help='whether to use jittering or not', action='store_true')
        parser.add_argument('--num_negatives', default=3, type=int,  help='Number of negative pairs in a training batch for 1 positive')
        parser.add_argument('--num_workers', default=8, type=int,  help='Number of workers to use. Max is 8.')
        parser.add_argument('--normalize_vis', default=False, type=bool, help='Whether to normalize vis features or not')
        parser.add_argument('--normalize_lang', default=True, type=bool, help='Whether to normalize language features or not')
        parser.add_argument('--scale_criterion', default=1.0, type=float, help='Scaling criterion for log-loss vanishing gradient')
        parser.add_argument('--l2norm_input', help='whether to L2 normalize precomp appearance features and language', action='store_true')
        parser.add_argument('--additional_neg_batch', default=500, type=int, help='Additional negatives to sample in batch')


        """ Evaluation """
        parser.add_argument('--nms_thresh', default=0.5 ,type=float, help='NMS threshold on proposals (used at test time). Candidates are already filtered nms 0.5')
        parser.add_argument('--epoch_model', default='best' ,type=str, help='At which epoch to load the model. Default is best. E.g. epoch50')
        parser.add_argument('--cand_test', default='candidates', type=str, help='Whether to evaluate on GT boxes or candidates')
        parser.add_argument('--subset_test', default='', type=str, help='Which subset to use for evaluation')
        parser.add_argument('--use_objscoreprecomp', help='Use s/o scores from object detector', action='store_true')


        
        """ Test aggreg """
        parser.add_argument('--sim_method', default='emb_word2vec', type=str, help='which similarity method to use')
        parser.add_argument('--thresh_method', default=None, type=str, help='whether to threshold the similarities')
        parser.add_argument('--alpha_r', default=0.5, type=float, help='Weight given to predicate similarity between source and target')
        parser.add_argument('--alpha_s', default=0.0, type=float, help='Weight given to subject similarity between source and target')
        parser.add_argument('--use_target', help='whether to use target triplet as source', action='store_true')
        parser.add_argument('--embedding_type', default='target', type=str, help='Embedding type')
        parser.add_argument('--minimal_mass', default=0, type=float, help='Minimal mass to use')


        """ Analogy """
        parser.add_argument('--use_analogy', help='Whether to use analogy transformation', action='store_true')
        parser.add_argument('--analogy_type', default='hybrid', type=str, help='type of analogy to use')
        parser.add_argument('--gamma', default='deep' ,type=str, help='Which gamma function to use for analogy making. The gamma function computes deformation')
        parser.add_argument('--lambda_reg', default=1, type=int,  help='Weight between regularization term and matching')
        parser.add_argument('--num_source_words_common', default=2, type=int,  help='Minimal number of words in source triplets common to target triplet')
        parser.add_argument('--restrict_source_object', help='Whether to restrict the source triplet to have same object', action='store_true')
        parser.add_argument('--restrict_source_subject', help='Whether to restrict the source triplet to have same subject', action='store_true')
        parser.add_argument('--restrict_source_predicate', help='Whether to restrict the source triplet to have same predicate', action='store_true')
        parser.add_argument('--normalize_source', help='Whether to L2 renormalize the source predictors after aggregation', action='store_true')
        parser.add_argument('--apply_deformation', help='Whether to apply deformation on source triplets', action='store_true')
        parser.add_argument('--precomp_vp_source_embedding', help='Whether to pre-computed vp embedding for source', action='store_true')
        parser.add_argument('--unique_source_random', help='If activated, will sample a unique source triplet at random among pre-selected ones', action='store_true')
        parser.add_argument('--detach_vis', help='Detach target visual visual phrase embedding in analogy branch', action='store_true')
        parser.add_argument('--detach_lang_analogy', help='Whether to detach language embedding in analogy transformation', action='store_true') 


        return parser


    def get_opts_from_dset(self, opt, dset):
        """ Load additional options from dataset object """
        
        opt.vocab_grams     = dset.vocab_grams
        opt.idx_sro_to      = dset.idx_sro_to
        opt.idx_to_vocab    = dset.idx_to_vocab
        opt.word_embeddings = dset.word_embeddings
        opt.d_appearance    = dset.d_appearance
        opt.occurrences     = dset.get_occurrences_precomp(opt.train_split)
        opt.classes         = dset.classes
        opt.predicates      = dset.predicates

        return opt


    def write_opts_dir(self, opt, logger_path):
        """ Write options in directory """

        f = open(osp.join(logger_path, "run_options.yaml"),"w")
        for key, val in vars(opt).iteritems():
            f.write("%s : %s\n" %(key,val))
        f.close()


    def get_res_dir(self, opt, name):
        """ Get results directory : opt.logger_dir/opt.exp_name/name """

        save_dir = osp.join(opt.logger_dir, opt.exp_name, name)

        if 'aggreg' in opt.embedding_type:

            sim_method      = opt.sim_method
            thresh_method   = opt.thresh_method
            use_target      = opt.use_target
            alpha_r         = opt.alpha_r
            alpha_s         = opt.alpha_s
            minimal_mass    = opt.minimal_mass


            sub_dir = 'sim-' + sim_method
            if alpha_r:
                sub_dir = sub_dir + '_' + 'alphar-' + str(alpha_r)
            if alpha_s:
                sub_dir = sub_dir + '_' + 'alphas-' + str(alpha_s)
            if thresh_method:
                sub_dir = sub_dir + '_' + 'tresh-' + thresh_method
            if use_target:
                sub_dir = sub_dir + '_' + 'usetarget'
            if minimal_mass > 0:
                sub_dir = sub_dir + '_' + 'mass-' + str(minimal_mass)

            save_dir = osp.join(save_dir, sub_dir)

        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        return save_dir




