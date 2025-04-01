// generate_music.js – compatible avec émotion + intensité
const mm = require('@magenta/music/node/music_vae');
const fs = require('fs');
const { write } = require('@magenta/music/node/midi_io');

// Mappage d'émotions vers des accords
const emotionMap = {
  "colère":     { chord: "C minor" },
  "tristesse":  { chord: "A minor" },
  "peur":       { chord: "D minor" },
  "joie":       { chord: "C major" },
  "doute":      { chord: "E minor" },
  "nostalgie":  { chord: "F major" }
};

const outputPath = process.argv[2];
const emotion = process.argv[3] || "joie";
const intensity = parseFloat(process.argv[4]) || 1.0;

const chord = emotionMap[emotion]?.chord || "C major";
const temperature = Math.min(Math.max(intensity, 0.1), 2.0); // entre 0.1 et 2.0

const model = new mm.MusicVAE('https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_2bar_small');

model.initialize().then(async () => {
  const z = await model.sample(1, temperature);
  let sequence = z[0];

  const qpm = 60 + (intensity * 40); // tempo entre 60 et 140 BPM
  sequence.tempos = [{ qpm }];
  sequence.quantizationInfo = { stepsPerQuarter: 4 };

  sequence.notes.forEach(n => {
    n.instrument = 0;
  });

  const midi = await write(sequence);
  fs.writeFileSync(outputPath, Buffer.from(midi));
});
