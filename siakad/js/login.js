document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("loginForm").addEventListener("submit", function(event) {
        event.preventDefault(); // Mencegah halaman reload

        let username = document.getElementById("pengguna").value;
        let password = document.getElementById("password").value;

        // Data login contoh (tanpa backend)
        let validUser = "admin";
        let validPass = "12345";

        if (username === "" || password === "") {
            alert("Mohon isi semua kolom!");
            return;
        }

        if (username === validUser && password === validPass) {
            alert("Login berhasil!");
            window.location.href = "index.html"; // Redirect setelah login
        } else {
            alert("Username atau password salah!");
        }
    });
});