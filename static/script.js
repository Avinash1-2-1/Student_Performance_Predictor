const CIRCUMFERENCE = 503;

const studyInput = document.getElementById('study_hours');
const attendanceInput = document.getElementById('attendance');
const previousInput = document.getElementById('previous_score');

const studyOut = document.getElementById('study_hours_out');
const attendanceOut = document.getElementById('attendance_out');
const previousOut = document.getElementById('previous_score_out');

studyInput.addEventListener('input', () => {
  studyOut.textContent = studyInput.value;
});
attendanceInput.addEventListener('input', () => {
  attendanceOut.textContent = `${attendanceInput.value}%`;
});
previousInput.addEventListener('input', () => {
  previousOut.textContent = previousInput.value;
});

const form = document.getElementById('predict-form');
const reportEmpty = document.getElementById('report-empty');
const reportFilled = document.getElementById('report-filled');
const reportError = document.getElementById('report-error');

const gradeRing = document.getElementById('grade-ring');
const gradeLetter = document.getElementById('grade-letter');
const scoreValue = document.getElementById('score-value');
const probStatus = document.getElementById('prob-status');
const probPercent = document.getElementById('prob-percent');
const probBarFill = document.getElementById('prob-bar-fill');
const remark = document.getElementById('remark');

function remarkFor(studyHours, attendance, previousScore, willPass) {
  if (!willPass) {
    if (studyHours < 10) return "More study hours would move this the most.";
    if (attendance < 65) return "Attendance is the weak link here.";
    return "Close — a bit more prep should tip this over.";
  }
  if (studyHours >= 20 && attendance >= 90) return "Strong habits across the board. Keep it up.";
  if (studyHours < 8) return "Passing, but study hours are doing the least work.";
  if (attendance < 70) return "Passing despite spotty attendance — don't push it.";
  return "Solid, consistent effort. On track.";
}

function resetVisuals() {
  gradeRing.style.strokeDashoffset = CIRCUMFERENCE;
  probBarFill.style.width = '0%';
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  reportError.hidden = true;
  resetVisuals();

  const payload = {
    study_hours: studyInput.value,
    attendance: attendanceInput.value,
    previous_score: previousInput.value,
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error('Prediction failed');
    const data = await res.json();

    reportEmpty.hidden = true;
    reportFilled.hidden = false;

    gradeLetter.textContent = data.grade;
    scoreValue.textContent = data.predicted_final_score;

    const pct = Math.round(data.pass_probability * 100);
    probPercent.textContent = `${pct}%`;
    probStatus.textContent = data.will_pass ? 'Likely to pass' : 'At risk of failing';

    remark.textContent = remarkFor(
      parseFloat(studyInput.value),
      parseFloat(attendanceInput.value),
      parseFloat(previousInput.value),
      data.will_pass
    );

    // Trigger animations on next frame so the transition actually fires
    requestAnimationFrame(() => {
      gradeRing.style.strokeDashoffset = CIRCUMFERENCE * (1 - data.pass_probability);
      probBarFill.style.width = `${pct}%`;
    });

  } catch (err) {
    reportEmpty.hidden = true;
    reportFilled.hidden = true;
    reportError.hidden = false;
  }
});
