#!/usr/bin/env python

"""Consumes stream for printing all messages to the console.
"""

import argparse
from datetime import datetime, timedelta
import json
import base64
import sys
import socket
import pandas as pd
from pyrtcm import RTCMMessage
from confluent_kafka import Consumer, KafkaError, KafkaException
from gnss_tec import tec, gnss
from check_tec import realtime_check_sat
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


df = pd.DataFrame(columns=('Timestamp', 'Satellite', 'Phase tec', 'P range tec'))
glob_sat = 'G08'
plt.ion()
fig_phase = plt.figure()
fig_range = plt.figure()


def gpsmsectotime(msec,leapseconds) -> datetime:
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.strptime("1980-01-06 00:00:00",datetimeformat)
    tdiff = datetime.utcnow() - epoch  + timedelta(seconds=(leapseconds - 19))
    gpsweek = tdiff.days // 7 
    t = epoch + timedelta(weeks=gpsweek) + timedelta(milliseconds=msec) - timedelta(seconds=(leapseconds - 19))
    return t


def msg_process(msg):
    global df

    val = msg.value()
    # dval = json.loads(val)

    # base64_payload = dval['bin_message']
    # rtcm_payload = base64.b64decode(base64_payload.encode('utf-8'))
    rtcm_msg = RTCMMessage(payload=val)

    if rtcm_msg.identity == '1004':
        delimiter = 299792.46
        speed_of_light = 299792458

        sat_id = rtcm_msg.DF009_01
        sat = f'G{sat_id:02}'

        p_range_1 = rtcm_msg.DF011_01 + delimiter * rtcm_msg.DF014_01
        p_range_2 = p_range_1 + rtcm_msg.DF017_01

        f1 = gnss.FREQUENCY.get('G').get(1)
        f2 = gnss.FREQUENCY.get('G').get(2)

        phase_range_1 = p_range_1 + rtcm_msg.DF012_01
        phase_range_2 = p_range_1 + rtcm_msg.DF018_01

        phase_1 = phase_range_1 / (speed_of_light / f1)
        phase_2 = phase_range_2 / (speed_of_light / f2)
        
        t = tec.Tec(datetime.now(), 'GPS', sat)
        t.p_range = {1: p_range_1, 2: p_range_2}
        t.p_range_code = {1: 'C1C', 2: 'C2W'}
        p_range_tec = t.p_range_tec

        t.phase = {1: phase_1, 2: phase_2}
        t.phase_code = {1: 'L1C', 2: 'L2S'}
        phase_tec = t.phase_tec

        timestamp = gpsmsectotime(rtcm_msg.DF004, 37)

        df_data = [timestamp, sat, phase_tec, p_range_tec]

        local_df = pd.DataFrame(data=[df_data], columns=('Timestamp', 'Satellite', 'Phase tec', 'P range tec'))

        df = pd.concat([df, local_df], ignore_index=True)

        # print(sat)
        # print(f'timestamp = {rtcm_msg.DF004}')
        # print(f'time = {timestamp}')
        # print(f'P range 1: original = {dval["P range 1"]}, rtcm = {p_range_1}')
        # print(f'P range 2: original = {dval["P range 2"]}, rtcm = {p_range_2}')
        # print(f'P range tec: original = {dval["P range tec"]}, rtcm = {p_range_tec}')
        # print(f'Phase 1: original = {dval["Phase 1"]}, rtcm = {phase_1}')
        # print(f'Phase 2: original = {dval["Phase 2"]}, rtcm = {phase_2}')
        # print(f'Phase tec: original = {dval["Phase tec"]}, rtcm = {phase_tec}')

        if sat == glob_sat:
            df_for_check = df[df['Satellite'] == sat]
            df_for_check = df_for_check[-120:]
            realtime_check_sat(df_for_check, fig_phase, fig_range, sat)
            plt.pause(0.0001)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topic', type=str,
                        help='Name of the Kafka topic to stream.')

    args = parser.parse_args()

    conf = {'bootstrap.servers': 'localhost:9092',
            'default.topic.config': {'auto.offset.reset': 'smallest'},
            'group.id': socket.gethostname()}

    consumer = Consumer(conf)

    running = True

    try:
        while running:
            consumer.subscribe([args.topic])

            msg = consumer.poll(1)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    sys.stderr.write('Topic unknown, creating %s topic\n' %
                                     (args.topic))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                msg_process(msg)

    except KeyboardInterrupt:
        pass

    finally:
        # Close down consumer to commit final offsets.
        consumer.close()
        plt.close(fig_phase)
        plt.close(fig_range)


if __name__ == "__main__":
    main()
