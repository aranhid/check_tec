#!/usr/bin/env python

"""Consumes stream for printing all messages to the console.
"""

import queue
import sys
import socket
import argparse
import threading
import pandas as pd

from pprint import pprint
from pyrtcm import RTCMMessage
from gnss_tec import tec, gnss
from datetime import datetime, timedelta
from confluent_kafka import Consumer, KafkaError, KafkaException

from check_tec import check_phase_tec, check_range_tec

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

last_timestamp = None
window_size = 3600
q = queue.Queue()


satellites_dataframe = pd.DataFrame()
problems_dataframe_phase = pd.DataFrame()
problems_dataframe_range = pd.DataFrame()


def gpsmsectotime(msec,leapseconds) -> datetime:
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.strptime("1980-01-06 00:00:00",datetimeformat)
    tdiff = datetime.utcnow() - epoch  + timedelta(seconds=(leapseconds - 19))
    gpsweek = tdiff.days // 7 
    t = epoch + timedelta(weeks=gpsweek) + timedelta(milliseconds=msec) - timedelta(seconds=(leapseconds - 19))
    return t


def msg_process(msg, topic):
    if not msg is None:
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition event
                sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                (msg.topic(), msg.partition(), msg.offset()))
            elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                sys.stderr.write('Topic unknown, creating %s topic\n' %
                                (topic))
            elif msg.error():
                raise KafkaException(msg.error())
        else:
            q.put(msg)


def update_sat_data(sat, timestamp, phase_tec, p_range_tec):
    global last_timestamp
    global satellites_dataframe

    df_data = [timestamp, sat, phase_tec, p_range_tec]

    local_df = pd.DataFrame(data=[df_data], columns=('Timestamp', 'Satellite', 'Phase tec', 'P range tec'))

    satellites_dataframe = pd.concat([satellites_dataframe, local_df], ignore_index=True)
    
    if timestamp != last_timestamp:
        last_timestamp = timestamp
        time_border = last_timestamp - timedelta(seconds=window_size)
        for sat in satellites_dataframe['Satellite'].unique():
            sat_df = satellites_dataframe[satellites_dataframe['Satellite'] == sat]
            sat_df_outstanding = sat_df[sat_df['Timestamp'] <= time_border]
            satellites_dataframe.drop(index=sat_df_outstanding.index, inplace=True)

    satellites_dataframe.to_csv("satellites_dataframe.csv")


def check_phase(sat):
    global problems_dataframe_phase
    df = satellites_dataframe[satellites_dataframe['Satellite'] == sat]
    phase_tec_problems = []

    checked_phase = check_phase_tec(df, std_mult=3.5)
    if not checked_phase.empty:
        checked_phase['Satellite'] = sat
        red_phase_tec = checked_phase[checked_phase['Color'] == 'red']
        phase_tec_problems = list(zip(red_phase_tec['Timestamp'].values, red_phase_tec['Phase tec'].values))

    if not problems_dataframe_phase.empty:
        prev_problems = problems_dataframe_phase[problems_dataframe_phase['Satellite'] == sat]
        problems_dataframe_phase.drop(index=prev_problems.index, inplace=True)

    if not checked_phase.empty:    
        problems_dataframe_phase = pd.concat([problems_dataframe_phase, checked_phase], ignore_index=True)

    problems_dataframe_phase.to_csv("problems_phase.csv")

    pprint(phase_tec_problems)


def check_range(sat):
    global problems_dataframe_range
    df = satellites_dataframe[satellites_dataframe['Satellite'] == sat]
    range_tec_problems = []

    checked_range = check_range_tec(df, std_mult=3.5)
    if not checked_range.empty:
        checked_range['Satellite'] = sat
        red_range_tec = checked_range[checked_range['Color'] == 'red']
        range_tec_problems = list(zip(red_range_tec['Timestamp'].values, red_range_tec['P range tec'].values))

    if not problems_dataframe_range.empty:
        prev_problems = problems_dataframe_range[problems_dataframe_range['Satellite'] == sat]
        problems_dataframe_range.drop(index=prev_problems.index, inplace=True)
    
    if not checked_range.empty:
        problems_dataframe_range = pd.concat([problems_dataframe_range, checked_range], ignore_index=True)

    problems_dataframe_range.to_csv("problems_range.csv")

    pprint(range_tec_problems)


def worker():
    while True:
        msg = q.get()
        val = msg.value()
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

            print(sat)
            update_sat_data(sat, timestamp, phase_tec, p_range_tec)
            check_phase(sat)
            check_range(sat)
            
        q.task_done()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', type=str, help='Host of the Kafka broker')
    parser.add_argument('topic', type=str, help='Name of the Kafka topic to stream.')

    args = parser.parse_args()

    host = args.host
    topic = args.topic

    conf = {'bootstrap.servers': f'{host}:9092',
            'default.topic.config': {'auto.offset.reset': 'smallest'},
            'group.id': socket.gethostname()}

    consumer = Consumer(conf)
    consumer.subscribe([topic])

    workers = []

    for i in range(1):
        wrkr = threading.Thread(target=worker, daemon=True)
        wrkr.start()
        workers.append(wrkr)

    try:
        while True:
            messages = consumer.consume(10, 1)
            for msg in messages:
                msg_process(msg, topic)

    except KeyboardInterrupt:
        pass

    finally:
        # Close down consumer to commit final offsets.
        consumer.close()
        q.join()


if __name__ == "__main__":
    main()
