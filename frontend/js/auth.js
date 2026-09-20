/* Shared Supabase auth helper. API calls carry the current access token. */
(function () {
    let clientPromise = null;

    function getClient() {
        if (!clientPromise) {
            clientPromise = fetch('/api/config')
                .then(function (res) {
                    if (res.ok) return res.json();
                    return res.json().then(function (body) { throw new Error(body.error || 'Auth is not configured'); });
                })
                .then(function (cfg) { return window.supabase.createClient(cfg.supabase_url, cfg.supabase_anon_key); });
            clientPromise.catch(function () { clientPromise = null; });
        }
        return clientPromise;
    }

    async function accessToken() {
        const client = await getClient();
        const result = await client.auth.getSession();
        return result.data.session ? result.data.session.access_token : null;
    }

    async function api(path, options) {
        options = options || {};
        const headers = new Headers(options.headers || {});
        const token = await accessToken();
        if (token) headers.set('Authorization', 'Bearer ' + token);
        return fetch(path, Object.assign({}, options, { headers: headers }));
    }

    async function hydrateImages(root) {
        const images = root.querySelectorAll('img[data-secure-src]');
        await Promise.all(Array.prototype.map.call(images, async function (img) {
            const res = await api(img.getAttribute('data-secure-src'));
            if (res.ok) img.src = URL.createObjectURL(await res.blob());
        }));
    }

    async function signIn(email, password) {
        const client = await getClient();
        return client.auth.signInWithPassword({ email: email, password: password });
    }

    async function signOut() {
        const client = await getClient();
        await client.auth.signOut();
    }

    window.VR = { api: api, hydrateImages: hydrateImages, signIn: signIn, signOut: signOut };
})();
