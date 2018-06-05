import copy
from pygaze.settings import settings
settings.DISPSIZE = (200, 200)
settings.DISPTYPE = 'pygame'
settings.MOUSEVISIBLE = True
# Eye tracker properties
# settings.TRACKERTYPE = 'opengaze'
settings.FULLSCREEN = False


from pygaze._eyetracker.libopengaze import OpenGazeTracker
from pygaze._eyetracker.baseeyetracker import BaseEyeTracker
from pygaze._eyetracker.opengaze import OpenGazeTracker as OpenGaze


from pygaze.screen import Screen
from pygaze.keyboard import Keyboard
from pygaze.sound import Sound
from pygaze.settings import settings

import socket
import time

try:
    from pygaze._misc.misc import copy_docstr
except:
    pass


class OpenGazeTrackerRETTNA(OpenGazeTracker):

    def __init__(self, display, logfile=settings.LOGFILE, \
                 eventdetection=settings.EVENTDETECTION, \
                 saccade_velocity_threshold=35, \
                 saccade_acceleration_threshold=9500, \
                 blink_threshold=settings.BLINKTHRESH, \
                 **args):

        """Initializes the OpenGazeTracker object

        arguments
        display	-- a pygaze.display.Display instance

        keyword arguments
        logfile	-- logfile name (string value); note that this is the
                   name for the eye data log file (default = LOGFILE)
        """

        # try to copy docstrings (but ignore it if it fails, as we do
        # not need it for actual functioning of the code)
        try:
            copy_docstr(BaseEyeTracker, OpenGazeTracker)
        except:
            # we're not even going to show a warning, since the copied
            # docstring is useful for code editors; these load the docs
            # in a non-verbose manner, so warning messages would be lost
            pass

        # object properties
        self.disp = display
        self.screen = Screen()
        self.dispsize = settings.DISPSIZE  # display size in pixels
        self.screensize = settings.SCREENSIZE  # display size in cm
        self.kb = Keyboard(keylist=['space', 'escape', 'q'], timeout=1)
        self.errorbeep = Sound(osc='saw', freq=100, length=100)

        # output file properties
        self.outputfile = logfile + '.tsv'
        self.extralogname = logfile + '_log.txt'
        self.extralogfile = open(self.extralogname, 'w')

        # eye tracker properties
        self.connected = False
        self.recording = False
        self.errdist = 2  # degrees; maximal error for drift correction
        self.pxerrdist = 30  # initial error in pixels
        self.maxtries = 100  # number of samples obtained before giving up (for obtaining accuracy and tracker distance information, as well as starting or stopping recording)
        self.prevsample = (-1, -1)
        self.prevps = -1

        # event detection properties
        self.fixtresh = 1.5  # degrees; maximal distance from fixation start (if gaze wanders beyond this, fixation has stopped)
        self.fixtimetresh = 100  # milliseconds; amount of time gaze has to linger within self.fixtresh to be marked as a fixation
        self.spdtresh = saccade_velocity_threshold  # degrees per second; saccade velocity threshold
        self.accthresh = saccade_acceleration_threshold  # degrees per second**2; saccade acceleration threshold
        self.blinkthresh = blink_threshold  # milliseconds; blink detection threshold used in PyGaze method
        self.eventdetection = eventdetection
        self.set_detection_type(self.eventdetection)
        self.weightdist = 10  # weighted distance, used for determining whether a movement is due to measurement error (1 is ok, higher is more conservative and will result in only larger saccades to be detected)

        # connect to the tracker
        self.opengaze = OpenGazeRETNNA(ip='192.168.0.32', port=4242, \
                                       logfile=self.outputfile, debug=True)

        # get info on the sample rate
        # TODO: Compute after streaming some samples?
        self.samplerate = 60.0
        self.sampletime = 1000.0 / self.samplerate

        # initiation report
        self._elog("pygaze initiation report start")
        self._elog("display resolution: %sx%s" % (self.dispsize[0], self.dispsize[1]))
        self._elog("display size in cm: %sx%s" % (self.screensize[0], self.screensize[1]))
        self._elog("samplerate: %.2f Hz" % self.samplerate)
        self._elog("sampletime: %.2f ms" % self.sampletime)
        self._elog("fixation threshold: %s degrees" % self.fixtresh)
        self._elog("speed threshold: %s degrees/second" % self.spdtresh)
        self._elog("acceleration threshold: %s degrees/second**2" % self.accthresh)
        self._elog("pygaze initiation report end")

    def sample(self):
        # Get newest sample.
        tracker = self.opengaze

        # If there is no current record yet, return None.
        tracker._inlock.acquire()

        if len(tracker._incoming_queue) == 0:
            frames = None
        else:
            frames = copy.deepcopy(tracker._incoming_queue)
            tracker._incoming_queue = []

        tracker._inlock.release()

        return frames

    def start_recording(self):
        super().start_recording()
        self.opengaze._incoming_queue = []


class OpenGazeRETNNA(OpenGaze):


    def _process_incoming(self):

        self._debug_print("Incoming Thread started.")
        self._incoming_queue = []

        while self._connected.is_set():

            # Lock the socket to prevent other Threads from simultaneously
            # accessing it.
            self._socklock.acquire()
            # Get new messages from the OpenGaze Server.
            timeout = False
            try:
                instring = self._sock.recv(self._maxrecvsize)
            except socket.timeout:
                timeout = True
            # Get a received timestamp.
            t = time.time()
            # Unlock the socket again.
            self._socklock.release()

            # Skip further processing if no new message came in.
            if timeout:
                self._debug_print("socket recv timeout")
                continue

            self._debug_print("Raw instring: %r" % (instring))

            # Split the messages (they are separated by '\r\n').
            messages = instring.decode().split('\r\n')

            # Check if there is currently an unfinished message.
            if self._unfinished:
                # Combine the currently unfinished message and the
                # most recent incoming message.
                messages[0] = copy.copy(self._unfinished) + messages[0]
                # Reset the unfinished message.
                self._unfinished = ''
            # Check if the last message was actually complete.
            if not messages[-1][-2:] == '/>':
                self._unfinished = messages.pop(-1)

            # Run through all messages.
            for msg in messages:
                self._debug_print("Incoming: %r" % (msg))
                # Parse the message.
                # print(msg)
                try:
                    command, msgdict = self._parse_msg(msg)
                except:
                    continue
                # Check if the incoming message is an acknowledgement.
                # Acknowledgements are also stored in a different dict,
                # which is used to monitor whether sent messages are
                # properly received.
                if command == 'ACK':
                    self._acklock.acquire()
                    self._acknowledgements[msgdict['ID']] = copy.copy(t)
                    self._acklock.release()
                # Acquire the Lock for the incoming dict, so that it
                # won't be accessed at the same time.
                self._inlock.acquire()
                # Check if this command is already in the current dict.
                if command not in self._incoming.keys():
                    self._incoming[command] = {}
                # Some messages have no ID, for example 'REC' messages.
                # We simply assign 'NO_ID' as the ID.
                if 'ID' not in msgdict.keys():
                    msgdict['ID'] = 'NO_ID'
                # Check if this ID is already in the current dict.
                if msgdict['ID'] not in self._incoming[command].keys():
                    self._incoming[command][msgdict['ID']] = {}
                # Add receiving time stamp, and the values for each
                # parameter to the current dict.
                self._incoming[command][msgdict['ID']]['t'] = \
                    copy.copy(t)
                for par, val in msgdict.items():
                    self._incoming[command][msgdict['ID']][par] = \
                        copy.copy(val)
                # Log sample if command=='REC' and when the logging
                # event is set.
                if command == 'REC' and self._logging.is_set():
                    self._logqueue.put(copy.deepcopy(self._incoming[command][msgdict['ID']]))
                # Unlock the incoming dict again.

                #Filling queue
                if 'REC' in self._incoming.keys() and 'NO_ID' in self._incoming['REC'].keys():
                    self._incoming_queue.append(self._incoming['REC']['NO_ID'])

                self._inlock.release()

        self._debug_print("Incoming Thread ended.")
        return