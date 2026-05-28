package com.example;

import com.example.util.Text;

public class Main {
    public void run() {
        String msg = Text.upper("hi");
        Handler h = new Handler(msg);
        h.greet();
        helper();
    }

    private void helper() {
    }
}
