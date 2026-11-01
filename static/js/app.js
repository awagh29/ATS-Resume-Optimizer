(function () {
  const form = document.getElementById("optimize-form");
  if (form) {
    const btn = document.getElementById("submit-btn");
    const hint = document.getElementById("progress-hint");
    const resume = document.getElementById("resume_latex");
    form.addEventListener("submit", function (e) {
      if (form.classList.contains("busy")) {
        e.preventDefault();
        return;
      }
      const jd = document.getElementById("job_description");
      const errors = [];
      if (!resume.value.trim() && !(document.getElementById("resume_file").files || []).length) {
        errors.push("Paste or upload a LaTeX resume.");
      }
      if (!jd.value.trim()) {
        errors.push("Paste a job description.");
      }
      if (errors.length) {
        e.preventDefault();
        alert(errors.join("\n"));
        return;
      }
      form.classList.add("busy");
      if (btn) { btn.textContent = "Working…"; btn.disabled = true; }
      if (hint) hint.hidden = false;
    });

    const fileInput = document.getElementById("resume_file");
    resume.addEventListener("input", function () {
      if (fileInput) fileInput.value = "";
    });
    if (fileInput) {
      fileInput.addEventListener("change", function () {
        const file = fileInput.files && fileInput.files[0];
        if (!file) return;
        if (!file.name.toLowerCase().endsWith(".tex")) {
          alert("Choose a .tex file containing your LaTeX source.");
          fileInput.value = "";
          return;
        }
        if (file.size > resume.maxLength * 4) {
          alert("This file is too large. Choose a smaller .tex resume.");
          fileInput.value = "";
          return;
        }
        const reader = new FileReader();
        reader.onload = function () {
          if (typeof reader.result === "string" && fileInput.files[0] === file) {
            resume.value = reader.result;
          }
        };
        reader.readAsText(file);
      });
    }

    window.addEventListener("pageshow", function (event) {
      if (event.persisted) window.location.reload();
    });

    function loadTemplate(id, targetId) {
      const tpl = document.getElementById(id);
      const target = document.getElementById(targetId);
      if (tpl && target) {
        target.value = tpl.content.textContent;
      }
    }

    const loadResume = document.getElementById("load-sample-resume");
    const loadJd = document.getElementById("load-sample-jd");
    if (loadResume) {
      loadResume.addEventListener("click", function () {
        if (fileInput) fileInput.value = "";
        loadTemplate("sample-resume-data", "resume_latex");
      });
    }
    if (loadJd) {
      loadJd.addEventListener("click", function () {
        loadTemplate("sample-jd-data", "job_description");
        const role = document.getElementById("role_title");
        const company = document.getElementById("company");
        if (role && !role.value) role.value = "Senior Backend Engineer";
        if (company && !company.value) company.value = "Northwind Analytics";
      });
    }
  }
})();
