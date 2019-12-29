import numpy as np

MILLISECONDS_IN_SECOND = 1000.0

REGRET_WINDOW_SIZE = 8

B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = './video_size_'


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        self.buffer_chunk_num=0
        self.buffer_chunk_alternable_size=[]
        self.buffer_chunk_bitrate=[]
        self.buffer_chunk_remain_time = []

        self.buffer_chunk_rebuff = []
        self.buffer_chunk_last_bitrate = []

        '''
        tmp=[]
        for i in range(BITRATE_LEVELS):
            tmp.append(0)
        for i in range(REGRET_WINDOW_SIZE):
            self.buffer_chunk_alternable_size.append(tmp)
            self.buffer_chunk_bitrate.append(0)
            self.buffer_chunk_remain_time.append(0)
        '''
        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_video_chunk(self, quality,next_or_regret):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        assert  next_or_regret >=0 and next_or_regret<=REGRET_WINDOW_SIZE
        regret_succeed=2

        assert next_or_regret <= self.buffer_chunk_num or next_or_regret==0

        if next_or_regret==0:
            regret_succeed=0
            video_chunk_size = self.video_size[quality][self.video_chunk_counter]

            # use the delivery opportunity in mahimahi
            delay = 0.0  # in ms
            video_chunk_counter_sent = 0  # in bytes

            while True:  # download video chunk over mahimahi
                throughput = self.cooked_bw[self.mahimahi_ptr] \
                             * B_IN_MB / BITS_IN_BYTE
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size:

                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    delay += fractional_time
                    self.last_mahimahi_time += fractional_time
                    assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                    video_chunk_counter_sent=video_chunk_size
                    break

                video_chunk_counter_sent += packet_payload
                #print("druation:",duration)
                delay += duration
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

            delay *= MILLISECONDS_IN_SECOND
            delay += LINK_RTT

        # add a multiplicative noise to the delay
            delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)
            return_delay=delay            

            # rebuffer time
            rebuf = np.maximum(delay - self.buffer_size, 0.0)

            # update the buffer
            if self.buffer_size-delay <0:
                self.buffer_size=0.0
                self.buffer_chunk_num=0
                self.buffer_chunk_alternable_size = []
                self.buffer_chunk_bitrate = []
                self.buffer_chunk_remain_time = []
                self.buffer_chunk_rebuff = []
                self.buffer_chunk_last_bitrate = []
            else:
                self.buffer_size-=delay
                for i in range(self.buffer_chunk_num):
                    delay-=self.buffer_chunk_remain_time[0]
                    if delay<0:
                        self.buffer_chunk_remain_time[0]=-delay
                        break
                    else:
                        self.buffer_chunk_remain_time.pop(0)
                        self.buffer_chunk_last_bitrate.pop(0)
                        self.buffer_chunk_rebuff.pop(0)
                        self.buffer_chunk_num-=1
                        self.buffer_chunk_bitrate.pop(0)
                        self.buffer_chunk_alternable_size.pop(0)

            # add in the new chunk
            self.buffer_chunk_num+=1
            self.buffer_size += VIDEO_CHUNCK_LEN

            new_video_chunk_sizes = []
            for i in range(BITRATE_LEVELS):
                new_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

            self.buffer_chunk_bitrate.append(quality)
            self.buffer_chunk_alternable_size.append( list(new_video_chunk_sizes))
            self.buffer_chunk_remain_time.append(VIDEO_CHUNCK_LEN)
            if len(self.buffer_chunk_bitrate) >1:
                self.buffer_chunk_last_bitrate.append(self.buffer_chunk_bitrate[-2])
            else:
                self.buffer_chunk_last_bitrate.append(None)
            self.buffer_chunk_rebuff.append(rebuf)

            # sleep if buffer gets too large
            sleep_time = 0
            if self.buffer_size > BUFFER_THRESH:
                # exceed the buffer limit
                # we need to skip some network bandwidth here
                # but do not add up the delay
                drain_buffer_time = self.buffer_size - BUFFER_THRESH
                sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                             DRAIN_BUFFER_SLEEP_TIME


                self.buffer_size -= sleep_time

                sleep_time_tmp=sleep_time
                for i in range(self.buffer_chunk_num):
                    sleep_time_tmp-=self.buffer_chunk_remain_time[0]
                    if sleep_time_tmp<0:
                        self.buffer_chunk_remain_time[0]=-sleep_time_tmp
                        break
                    else:
                        self.buffer_chunk_remain_time.pop(0)
                        self.buffer_chunk_num-=1
                        self.buffer_chunk_bitrate.pop(0)
                        self.buffer_chunk_alternable_size.pop(0)
                        self.buffer_chunk_rebuff.pop(0)
                        self.buffer_chunk_last_bitrate.pop(0)


                while True:
                    duration = self.cooked_time[self.mahimahi_ptr] \
                               - self.last_mahimahi_time
                    if duration > sleep_time / MILLISECONDS_IN_SECOND:
                        self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                        break
                    sleep_time -= duration * MILLISECONDS_IN_SECOND
                    self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                    self.mahimahi_ptr += 1

                    if self.mahimahi_ptr >= len(self.cooked_bw):
                        # loop back in the beginning
                        # note: trace file starts with time 0
                        self.mahimahi_ptr = 1
                        self.last_mahimahi_time = 0

            # the "last buffer size" return to the controller
            # Note: in old version of dash the lowest buffer is 0.
            # In the new version the buffer always have at least
            # one chunk of video
            return_buffer_size = self.buffer_size

            self.video_chunk_counter += 1
            video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

            #chance to fix low quality chunk
            regret_chunk_bitrate=[]
            regret_chunk_remain_time=[]
            regret_chunk_alternate_size=[]
            for i in range(REGRET_WINDOW_SIZE):
                if self.buffer_chunk_num-REGRET_WINDOW_SIZE+i<0:
                    pass
                else:
                    regret_chunk_alternate_size.append(list(self.buffer_chunk_alternable_size[self.buffer_chunk_num-REGRET_WINDOW_SIZE+i]))
                    regret_chunk_bitrate.append(self.buffer_chunk_bitrate[self.buffer_chunk_num-REGRET_WINDOW_SIZE+i])
                    regret_chunk_remain_time.append(self.buffer_chunk_remain_time[self.buffer_chunk_num-REGRET_WINDOW_SIZE+i]/MILLISECONDS_IN_SECOND)

            for i in range(REGRET_WINDOW_SIZE-len(regret_chunk_alternate_size)):
                regret_chunk_alternate_size.append([0, 0, 0, 0, 0, 0])
                regret_chunk_bitrate.append(0)
                regret_chunk_remain_time.append(0)

            end_of_video = False
            if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
                end_of_video = True
                self.buffer_size = 0
                self.video_chunk_counter = 0


                self.buffer_chunk_num=0
                self.buffer_chunk_remain_time=[]
                self.buffer_chunk_alternable_size=[]
                self.buffer_chunk_bitrate=[]
                self.buffer_chunk_rebuff=[]
                self.buffer_chunk_last_bitrate=[]

                # pick a random trace file
                self.trace_idx = np.random.randint(len(self.all_cooked_time))
                self.cooked_time = self.all_cooked_time[self.trace_idx]
                self.cooked_bw = self.all_cooked_bw[self.trace_idx]

                # randomize the start point of the video
                # note: trace file starts with time 0
                self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

            next_video_chunk_sizes = []
            for i in range(BITRATE_LEVELS):
                next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

            return return_delay, \
                sleep_time, \
                return_buffer_size / MILLISECONDS_IN_SECOND, \
                rebuf / MILLISECONDS_IN_SECOND, \
                video_chunk_size, \
                next_video_chunk_sizes, \
                end_of_video, \
                video_chunk_remain, \
                regret_chunk_remain_time,regret_chunk_bitrate,regret_chunk_alternate_size,\
                video_chunk_counter_sent, \
                regret_succeed, \
                0,\
                None,None,0,self.buffer_chunk_num
        else:
            if self.buffer_chunk_num<=REGRET_WINDOW_SIZE:
                index=next_or_regret-1
            else:
                index=self.buffer_chunk_num-REGRET_WINDOW_SIZE+next_or_regret-1

            if index==self.buffer_chunk_num-1:
                latest_chunk_updated=1
            else:
                latest_chunk_updated=0
            back_lenth=self.buffer_chunk_num-1-index

            if self.video_size[quality][self.video_chunk_counter-1-(self.buffer_chunk_num-index-1)]!=self.buffer_chunk_alternable_size[index][quality]:
                print("env wrong!!!!!!!!!!!!!!!!check again!!!!!!!!!!!!!!!!!!!")
                print(self.video_chunk_counter-1-(self.buffer_chunk_num-index-1))
                print("index:",index)
                print("buffer num",self.buffer_chunk_num)
                print("self.vide_count",self.video_chunk_counter)
            assert self.video_size[quality][self.video_chunk_counter-1-(self.buffer_chunk_num-index-1)]==self.buffer_chunk_alternable_size[index][quality]


            video_chunk_size = self.video_size[quality][self.video_chunk_counter-1-(self.buffer_chunk_num-index-1)] \
				- self.video_size[self.buffer_chunk_bitrate[index]][self.video_chunk_counter-1-(self.buffer_chunk_num-index-1)]


            time_before_chunk=0
            for i in range(index):
                time_before_chunk+=self.buffer_chunk_remain_time[i]/MILLISECONDS_IN_SECOND

            total_time_available=time_before_chunk
            #when are we gonna stop???
            delay = 0.0  # in ms
            video_chunk_counter_sent = 0  # in bytes

            #print("\n\ntime before chunk:",time_before_chunk,"\n\n")


            while True:  # download video chunk over mahimahi
                throughput = self.cooked_bw[self.mahimahi_ptr] \
                             * B_IN_MB / BITS_IN_BYTE
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time

                if time_before_chunk<duration:
                    duration=time_before_chunk
                    time_before_chunk=0
                else:
                    time_before_chunk-=duration


                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size:
                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    video_chunk_counter_sent=video_chunk_size
                    delay += fractional_time
                    self.last_mahimahi_time += fractional_time
                    assert (self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                    regret_succeed=1
                    break

                if duration<self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time:
                    self.last_mahimahi_time +=duration
                else:
                    # print("druation:",duration)
                    self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                    self.mahimahi_ptr += 1

                delay += duration
                video_chunk_counter_sent += packet_payload



                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

                if time_before_chunk<=0:
                    regret_succeed=2
                    break

            old_bitrate=self.buffer_chunk_bitrate[index]
            last_bitrate=self.buffer_chunk_last_bitrate[index]
            old_rebuff=self.buffer_chunk_rebuff[index]
            if index<self.buffer_chunk_num-1:
                next_bitrate = self.buffer_chunk_bitrate[index + 1]
            else:
                next_bitrate=None


            if regret_succeed==2:
                delay=total_time_available
            else:
                self.buffer_chunk_bitrate[index] = quality
                if index<self.buffer_chunk_num-1:
                    self.buffer_chunk_last_bitrate[index+1]=quality

            if delay!=0:
                delay *= MILLISECONDS_IN_SECOND
                if regret_succeed!=2:
                    delay += LINK_RTT

            # add a multiplicative noise to the delay
            if regret_succeed!=2:
                delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

            return_delay=delay

            if self.buffer_size-delay <0:
                self.buffer_size=0.0
                self.buffer_chunk_num=0
                self.buffer_chunk_alternable_size = []
                self.buffer_chunk_bitrate = []
                self.buffer_chunk_remain_time = []
            else:
                self.buffer_size-=delay
                for i in range(self.buffer_chunk_num):
                    delay-=self.buffer_chunk_remain_time[0]
                    if delay<0:
                        self.buffer_chunk_remain_time[0]=-delay
                        break
                    else:
                        self.buffer_chunk_remain_time.pop(0)
                        self.buffer_chunk_num-=1
                        self.buffer_chunk_bitrate.pop(0)
                        self.buffer_chunk_alternable_size.pop(0)
                        self.buffer_chunk_rebuff.pop(0)
                        self.buffer_chunk_last_bitrate.pop(0)


            # chance to fix low quality chunk
            regret_chunk_bitrate = []
            regret_chunk_remain_time = []
            regret_chunk_alternate_size = []
            for i in range(REGRET_WINDOW_SIZE):
                if self.buffer_chunk_num - REGRET_WINDOW_SIZE + i < 0:
                    pass
                else:
                    regret_chunk_alternate_size.append(
                        list(self.buffer_chunk_alternable_size[self.buffer_chunk_num - REGRET_WINDOW_SIZE + i]))
                    regret_chunk_bitrate.append(self.buffer_chunk_bitrate[self.buffer_chunk_num - REGRET_WINDOW_SIZE + i])
                    regret_chunk_remain_time.append(self.buffer_chunk_remain_time[self.buffer_chunk_num - REGRET_WINDOW_SIZE + i] / MILLISECONDS_IN_SECOND)

            for i in range(REGRET_WINDOW_SIZE-len(regret_chunk_alternate_size)):
                regret_chunk_alternate_size.append([0, 0, 0, 0, 0, 0])
                regret_chunk_bitrate.append(0)
                regret_chunk_remain_time.append(0)

            return_buffer_size=self.buffer_size

            return return_delay, \
                   0, \
                   return_buffer_size / MILLISECONDS_IN_SECOND, \
                   old_rebuff/MILLISECONDS_IN_SECOND, \
                   video_chunk_size, \
                   0, \
                   0, \
                   0,\
                   regret_chunk_remain_time, regret_chunk_bitrate, regret_chunk_alternate_size, \
                   video_chunk_counter_sent, \
                   regret_succeed, \
                   old_bitrate,\
                   last_bitrate,next_bitrate,latest_chunk_updated,self.buffer_chunk_num





