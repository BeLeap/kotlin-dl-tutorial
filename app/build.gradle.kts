plugins {
  kotlin("jvm") version "1.5.31"
  application
}

repositories { mavenCentral() }

dependencies {
  implementation(platform("org.jetbrains.kotlin:kotlin-bom"))
  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
  testImplementation("org.jetbrains.kotlin:kotlin-test")
  testImplementation("org.jetbrains.kotlin:kotlin-test-junit")

  implementation("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.4.0")
}

application { mainClass.set("codes.beleap.kotlin_dl_tutorial.AppKt") }
