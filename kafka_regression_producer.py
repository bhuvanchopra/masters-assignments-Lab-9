from kafka import KafkaProducer
import datetime, time, random, threading


rates = [1, 10, 100] # messages per second
sample_sigma = 500
fuzz_sigma = 10


# the values we hope to reconstruct
m = 0.0
b = 0.0


def data_point_gauss(rand):
    """
    Return a random x,y data point near our data line.
    """
    x = rand.gauss(0, sample_sigma)
    y = m*x + b + rand.gauss(0, fuzz_sigma)
    return x, y


def send_at(rate):
    rand = random.Random() # docs say threadsafe, but construct one per thread anyway
    producer = KafkaProducer(bootstrap_servers=['199.60.17.210:9092', '199.60.17.193:9092'])
    topic = 'xy-' + str(rate)
    interval = 1/rate
    while True:
        x, y = data_point_gauss(rand)
        msg = '%s %s' % (x, y)
        producer.send(topic, msg.encode('ascii'))
        time.sleep(interval)

        
if __name__ == "__main__":
    for rate in rates:
        server_thread = threading.Thread(target=send_at, args=(rate,))
        server_thread.setDaemon(True)
        server_thread.start()

    while 1:
        time.sleep(1)
