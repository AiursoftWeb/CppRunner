(() => {
    const errorBox = document.getElementById('save-code-error');
    const errorMessage = errorBox.querySelector('.alert-message');
    const labels = document.getElementById('save-code-text');
    const text = (key, fallback) => labels.querySelector(`[data-key="${key}"]`)?.textContent.trim() || fallback;

    saveCodeModalEl.addEventListener('show.bs.modal', () => {
        errorBox.classList.add('d-none');
        codeTitleInput.classList.remove('is-invalid');
        codeTitleInput.removeAttribute('aria-invalid');
    });
    codeTitleInput.addEventListener('input', () => {
        codeTitleInput.classList.remove('is-invalid');
        codeTitleInput.removeAttribute('aria-invalid');
    });

    btnConfirmSave.addEventListener('click', async () => {
        if (btnConfirmSave.disabled) return;
        errorBox.classList.add('d-none');
        const title = codeTitleInput.value.trim();
        if (!title) {
            codeTitleInput.classList.add('is-invalid');
            codeTitleInput.setAttribute('aria-invalid', 'true');
            codeTitleInput.focus();
            return;
        }

        const saveResult = saveResultSwitch.checked;
        const formData = new FormData();
        formData.append('title', title);
        formData.append('code', editor.getValue());
        formData.append('language', langSelector.value);
        formData.append('isPublic', isPublicSwitch.checked);
        formData.append('saveResult', saveResult);
        if (saveResult) {
            formData.append('result', outputEditor.getValue());
            formData.append('error', errorEditor.getValue());
            if (lastResultCode !== null) formData.append('resultCode', lastResultCode);
        }
        if (currentIdToSave) formData.append('id', currentIdToSave);
        formData.append('__RequestVerificationToken', document.querySelector('input[name="__RequestVerificationToken"]').value);

        btnConfirmSave.disabled = true;
        btnConfirmSave.setAttribute('aria-busy', 'true');
        btnConfirmSave.textContent = text('saving', 'Saving...');
        try {
            const response = await fetch('/Codes/Save', { method: 'POST', body: formData });
            if (!response.ok || response.redirected) throw new Error('Save failed');
            const saved = await response.json();
            if (!Number.isInteger(saved.id) || saved.id <= 0) throw new Error('Invalid save response');
            window.location.href = '/Codes/Index';
        } catch {
            errorMessage.textContent = text('failed', 'Could not save. Please try again.');
            errorBox.classList.remove('d-none');
        } finally {
            btnConfirmSave.disabled = false;
            btnConfirmSave.removeAttribute('aria-busy');
            btnConfirmSave.textContent = text('save', 'Save');
        }
    });
})();
