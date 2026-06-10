const state = {
  task: null,
  startedAt: performance.now(),
  events: []
};

const videoBindings = [
  ['sourceVideo', 'source'],
  ['maskVideo', 'mask_overlay'],
  ['previewVideo', 'preview'],
  ['finalAVideo', 'final_a'],
  ['finalBVideo', 'final_b']
];

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function loadTask(task) {
  state.task = task;
  state.startedAt = performance.now();
  state.events = [];
  setText('taskMeta', `${task.study_id} / ${task.task_id} / ${task.condition}`);
  setText('promptText', task.prompt || 'No prompt supplied.');
  for (const [elementId, key] of videoBindings) {
    const element = document.getElementById(elementId);
    element.removeAttribute('src');
    if (task[key] && task[key].uri) {
      element.src = task[key].uri;
      element.title = task[key].label || key;
    }
  }
  renderEvents();
}

function renderEvents() {
  const list = document.getElementById('eventLog');
  list.innerHTML = '';
  for (const event of state.events) {
    const item = document.createElement('li');
    item.textContent = `${event.name} at ${event.elapsed_s.toFixed(3)}s`;
    list.appendChild(item);
  }
}

function value(id) {
  const node = document.getElementById(id);
  if (node.type === 'checkbox') return node.checked;
  return node.value;
}

function selectedPreference() {
  const selected = document.querySelector('input[name="preference"]:checked');
  return selected ? selected.value : null;
}

function exportAnnotation() {
  if (!state.task) throw new Error('No task loaded');
  const now = performance.now();
  const annotation = {
    schema_version: 'preview-ladder-human-eval/v0.1',
    study_id: state.task.study_id,
    assignment_id: state.task.assignment_id,
    task_id: state.task.task_id,
    condition: state.task.condition,
    response: {
      preview_trust: {
        structural_commitment_score: Number(value('trustStructural')),
        approve_preview_likelihood: Number(value('approvePreview')),
        expected_drift_risk: Number(value('expectedDriftRisk'))
      },
      pairwise_final_preference: {
        preference: selectedPreference(),
        rationale: value('rationale')
      },
      interaction_events: state.events,
      quality_controls: {
        instruction_check_answer: value('instructionCheck'),
        played_preview: value('playedPreview'),
        played_finals: value('playedFinals'),
        elapsed_total_s: (now - state.startedAt) / 1000
      }
    },
    client_metadata: {
      user_agent: navigator.userAgent,
      exported_at_unix_ms: Date.now()
    }
  };
  document.getElementById('output').textContent = JSON.stringify(annotation, null, 2);
}

for (const input of document.querySelectorAll('input[type="range"]')) {
  input.addEventListener('input', () => {
    const display = document.querySelector(`[data-value-for="${input.id}"]`);
    display.textContent = input.value;
  });
}

for (const button of document.querySelectorAll('[data-event]')) {
  button.addEventListener('click', () => {
    state.events.push({
      name: button.dataset.event,
      elapsed_s: (performance.now() - state.startedAt) / 1000
    });
    renderEvents();
  });
}

document.getElementById('loadSample').addEventListener('click', async () => {
  const response = await fetch('sample_task.json');
  loadTask(await response.json());
});

document.getElementById('taskFile').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  loadTask(JSON.parse(await file.text()));
});

document.getElementById('exportJson').addEventListener('click', () => {
  try {
    exportAnnotation();
  } catch (error) {
    document.getElementById('output').textContent = String(error);
  }
});
