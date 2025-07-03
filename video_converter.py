import subprocess

class VideoConverter:
    def __init__(self, input_file, output_file, qscale="0", ar="22050", r="60"):
        self.input_file = input_file
        self.output_file = output_file
        self.qscale = qscale
        self.ar = ar
        self.r = r

    def convert(self):
        ffmpeg_command = [
            "ffmpeg",
            "-i", self.input_file,
            "-qscale:v", self.qscale,      # explicitly video quality
            "-ar", self.ar,                # audio sampling rate
            "-r", self.r,                  # frame rate
            self.output_file
        ]
        
        try:
            subprocess.run(ffmpeg_command, check=True)
            print("Conversion completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Conversion failed: {e}")