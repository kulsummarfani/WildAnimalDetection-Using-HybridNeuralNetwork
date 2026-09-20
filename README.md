# Wild Animal Detection

Run the detector on a supplied sample image:

```bash
.venv/bin/python Main.py testImages/9.jpeg
```

Use any image, video, directory, stream URL, or camera index as the argument. The annotated image output is written to `output.png`; the command prints `1` when an animal is detected and `0` otherwise.

Each YOLO crop is also classified as Cheetah, Jaguar, Leopard, Lion, or Tiger using `model/vgg_bilstm_weights.hdf5`. To only run YOLO detection:

```bash
.venv/bin/python Main.py testImages/9.jpeg --no-classify
```

Grad-CAM evidence overlays are written to `gradcam/`. Lions, tigers, leopards, and jaguars alert at 60% classifier confidence; cheetahs alert at 85%. Disable overlays with `--no-gradcam`.

SMS alerts are disabled by default. To enable them, set Fast2SMS credentials and opt in explicitly:

```bash
export FAST2SMS_API_KEY='your-key'
export ALERT_PHONE_NUMBER='9999999999'
.venv/bin/python Main.py testImages/9.jpeg --send-alerts
```

Alternatively, create a `.env` file in the project root:

```bash
FAST2SMS_API_KEY=your-key
ALERT_PHONE_NUMBER=9999999999
```

Check that the values were read without sending a message:

```bash
.venv/bin/python Main.py --check-alert-config
```

The project is configured for Python 3.11. To recreate the environment:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Web dashboard

The integrated VanRakshak AI frontend is served by the Flask app. Start it
from the project root after installing the requirements:

```bash
.venv/bin/python webapp.py
```

Open `http://localhost:5000`. The dashboard accepts an uploaded frame, runs
the project detector and VGG-Bi-LSTM crop classifier, stores alert history in
Supabase, and serves protected annotated and Grad-CAM evidence images.
Movement and distance remain single-frame estimates; the Bi-LSTM model is not
fed a temporal sequence by this upload workflow.

### Supabase setup

1. Run [supabase_schema.sql](supabase_schema.sql) in the Supabase SQL editor.
2. Create users in Supabase Authentication and promote their roles using the
	SQL comments at the bottom of the schema file.
3. Disable public sign-ups in the Supabase Auth settings.
4. Add these server-only values to `.env`:

```text
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=<publishable-or-anon-key>
SUPABASE_SERVICE_KEY=<service-role-or-secret-key>
```

The service key stays in Flask and is never sent to the browser. Residents can
view alerts; watchmen and admins can run detection. Web-triggered SMS remains
opt-in with `SEND_SMS_ALERTS=1`. By default, the same species can trigger at
most one SMS every five minutes; set `SMS_COOLDOWN_SECONDS=0` to disable the
cooldown or choose another interval.
