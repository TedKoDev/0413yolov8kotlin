package com.example.a0413yolov8kotlin

import android.graphics.RectF
import java.io.File
import java.io.FileNotFoundException

data class Result(val classIndex: Int, val score: Float, val rectF: RectF)